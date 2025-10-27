from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GaugeField(nn.Module):
    """Stationary gauge field W_psi(x; y_embed)."""
    def __init__(self, in_ch_img: int = 1, in_ch_meas: int = 1, hidden: int = 64, depth: int = 4):
        super().__init__()
        assert in_ch_img >= 1 and in_ch_meas >= 0
        self.in_ch_img = int(in_ch_img)
        self.in_ch_meas = int(in_ch_meas)

        ch = hidden
        inc = self.in_ch_img + self.in_ch_meas
        layers = [nn.Conv2d(inc, ch, 3, 1, 1), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU()]
        # output channels == image channels (so h(x,t,y) has same shape as x)
        layers += [nn.Conv2d(ch, self.in_ch_img, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def _cat_xy(self, x: torch.Tensor, y_embed: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B,Cx,H,W), y_embed: (B,Cy,H,W) or None
        B, Cx, H, W = x.shape
        assert Cx == self.in_ch_img, f"Expected x with {self.in_ch_img} channels, got {Cx}"
        if self.in_ch_meas == 0:
            return x
        if y_embed is None:
            # pad zeros to match expected measurement channels
            y_pad = torch.zeros(B, self.in_ch_meas, H, W, device=x.device, dtype=x.dtype)
            return torch.cat([x, y_pad], dim=1)
        # resize spatially if needed
        if y_embed.shape[-2:] != (H, W):
            y_embed = F.interpolate(y_embed, size=(H, W), mode="bilinear", align_corners=False)
        # if channels mismatch, pad or crop
        Cy = y_embed.shape[1]
        if Cy < self.in_ch_meas:
            pad = torch.zeros(B, self.in_ch_meas - Cy, H, W, device=x.device, dtype=x.dtype)
            y_embed = torch.cat([y_embed, pad], dim=1)
        elif Cy > self.in_ch_meas:
            y_embed = y_embed[:, :self.in_ch_meas]
        return torch.cat([x, y_embed], dim=1)

    def forward(self, x: torch.Tensor, y_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = self._cat_xy(x, y_embed)
        return self.net(inp)


class GaugeFlow(nn.Module):
    """
    h_t^Y(x) = x + α(t) W(x;Y), with α(0)=α(1)=0 (endpoint identity).
    dt_h(x,t,Y) = α'(t) W(x;Y)         (exact derivative)
    jvp_h(x,t,Y,v) = v + α(t) D W(x;Y) v
    h_inv is solved by fixed-point iteration: x = z - α W(x;Y)

    Notes:
      - `max_step` scales α(t). Choose it small enough to ensure contraction: max_t α(t)*Lip(W) < 1.
      - For better diffeo guarantees, consider spectral normalization on W's convs.
    """
    def __init__(self, W: GaugeField, bump: str = "cosine",
                 max_step: float = 1.0, inv_iters: int = 3, **kwargs):
        super().__init__()
        _ = kwargs  # ignore any legacy extras
        self.W = W
        self.max_step = float(max_step)
        self.inv_iters = int(inv_iters)
        self.bump_type = str(bump)

    # ----- bump α(t) and its exact derivative α'(t) -----

    def _alpha_base(self, t: torch.Tensor) -> torch.Tensor:
        """
        Smooth bump in [0,1] with α(0)=α(1)=0. We use:
            α_base(t) = 2 * (1 - cos(pi t)) * t * (1 - t)
        which is C^1 and zero at the endpoints.
        """
        return 2.0 * (1.0 - torch.cos(torch.pi * t)) * t * (1.0 - t)

    def _alpha_base_dot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Exact derivative:
            d/dt α_base(t) = 2*(1-2t)*(1 - cos(pi t)) + 2*pi*sin(pi t)*t*(1 - t)
        """
        return 2.0 * (1.0 - 2.0 * t) * (1.0 - torch.cos(torch.pi * t)) \
             + 2.0 * torch.pi * torch.sin(torch.pi * t) * t * (1.0 - t)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.max_step * self._alpha_base(t)

    def _alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return self.max_step * self._alpha_base_dot(t)

    # ----- core maps -----

    def h(self, x: torch.Tensor, t: torch.Tensor, y_embed: Optional[torch.Tensor]) -> torch.Tensor:
        alpha = self._alpha(t)
        return x + alpha * self.W(x, y_embed)

    def dt_h(self, x: torch.Tensor, t: torch.Tensor, y_embed: Optional[torch.Tensor]) -> torch.Tensor:
        # exact ∂_t h = α'(t) W(x;Y); W has no explicit t-dependence
        alpha_dot = self._alpha_dot(t)
        return alpha_dot * self.W(x, y_embed)

    def jvp_h(self, x: torch.Tensor, t: torch.Tensor, y_embed: Optional[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        """
        Jacobian-vector product for h at (x,t): Dh(x) v = v + α(t) D W(x;Y) v.
        Uses forward-mode JVP on W; α(t) depends only on t, so its JVP wrt x is zero.
        """
        # Detach x from any upstream graph; we only need grads into W's parameters.
        x_req = x.detach().requires_grad_(True)
        alpha = self._alpha(t)
        # jvp of W at x in the direction v
        jvp = torch.autograd.functional.jvp(lambda xx: self.W(xx, y_embed), x_req, v,
                                            create_graph=True, strict=False)[1]
        return v + alpha * jvp

    def h_inv(self, z: torch.Tensor, t: torch.Tensor, y_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Fixed-point iterations for inverse of x -> x + α W(x;Y):
            x_{k+1} = z - α W(x_k;Y)
        Converges if α * Lip(W) < 1. Use small `max_step` or constrain W's Lipschitz.
        """
        alpha = self._alpha(t)
        x = z
        for _ in range(self.inv_iters):
            x = z - alpha * self.W(x, y_embed)
        return x
