from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple

from training.losses import fm_loss, gfm_loss, gfm_target
from models.posterior_mean import PosteriorMean
from models.rectified_flow import RectifiedFlow
from models.gauge import GaugeField, GaugeFlow




def sample_t(batch: int, device: torch.device) -> torch.Tensor:
    """Sample t ~ U[0,1] shaped for broadcasting over BCHW."""
    return torch.rand(batch, 1, 1, 1, device=device)


def y_embed_default(y: torch.Tensor) -> torch.Tensor:
    """Use degraded input as measurement embedding by default."""
    return y


class ENGRFAbs(nn.Module):
    """
    ENGRF (Endpoint-Neutral Gauge Rectified Flow) wrapper.

    Stage 0: train f_psi (posterior mean) supervised.
    Stage 1: train baseline RF via FM on Z_t = (1-t) X* + t X, target Δ = X - X*.
    Stage 2: train gauge (and optionally fine-tune RF) via gauged FM on
             tilde Z_t = h_t^Y(Z_t) with target ∂_t h_t^Y(Z_t) + (D h_t^Y)(Z_t) Δ,
             and prediction \tilde v(x) = ∂_t h_t^Y(x) + (D h_t^Y)(x) v(h_t^{-1}(x), t).
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg["model"]
        # Posterior mean backbone f_psi: y -> x*
        if mcfg["posterior_mean"] == 'swinir':
            cfg_name = "pm_swinir"
        else:
            cfg_name = "pm_unet"
            
        self.pm = PosteriorMean(mcfg["posterior_mean"], mcfg[cfg_name])
        self.pmrf_sigma_s = mcfg.get("pmrf_sigma_s", 0.03)
        self.pmrf_only = False
        # Baseline rectified flow v_theta
        self.rf = RectifiedFlow(**mcfg["rf_unet"])
        # Gauge: stationary field W_psi and its induced flow h_t^Y
        self.W = GaugeField(**mcfg["gauge_field"])
        self.hflow = GaugeFlow(self.W, **mcfg["gauge_flow"])  # exposes h, dt_h, jvp_h, h_inv

    # -------------------------- Shared helpers --------------------------- #
    @torch.no_grad()
    def check_h_identity(self, device: torch.device | None = None, tol: float = 1e-5):
       """Numerical sanity: h_t^Y = Id at t=0 and t=1."""
       device = device or self._device()
       # Small random tensor mimicking an image batch shape;
       # alternatively, sample from a real batch in the caller.
       B, C, H, W = 2, 1, 32, 32
       x = torch.randn(B, C, H, W, device=device)
       y = torch.randn(B, C, H, W, device=device)
       y_emb = y_embed_default(y)
       x0 = self.hflow.h(x, torch.zeros(B,1,1,1, device=device), y_emb)
       x1 = self.hflow.h(x, torch.ones(B,1,1,1, device=device),  y_emb)
       e0 = (x0 - x).abs().max().item()
       e1 = (x1 - x).abs().max().item()
       if not (e0 <= tol and e1 <= tol):
           raise RuntimeError(f"h identity violation: max|h(x,0)-x|={e0:.2e}, max|h(x,1)-x|={e1:.2e}")


    @torch.no_grad()
    def sample_pmrf(self, y: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """
        PMRF sampler:
        x0 = f_psi(y) + sigma_s * N(0,I)
        dx/dt = v_theta(x,t)
        Euler with midpoints (optional).
        """
        device = self._device()
        dtype  = y.dtype
        self.eval(); self.pm.eval()

        x0 = self.pm(y.to(device)).to(dtype)
        if self.pmrf_sigma_s > 0:
            x0 = x0 + self.pmrf_sigma_s * torch.randn_like(x0)

        x_t = x0
        dt  = 1.0 / float(steps)

        for i in range(steps):
            t_mid = torch.full((x_t.size(0),1,1,1), (i + 0.5) * dt, device=device, dtype=dtype)
            v = self.rf(x_t, t_mid).to(dtype)
            x_t = x_t + v * dt

        return x_t
        
    @torch.no_grad()
    def _sample_engrf(self, y: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """
        ENGRF deterministic sampler (tied-conjugacy ODE):
            dx/dt = ∂_t h_t^Y(x) + (D h_t^Y)(x) v_theta(h_t^{-1}(x), t)
        Starts at x*(y) because h_0^Y = Id. Euler integration with midpoints.
        """
        device = self._device()
        dtype  = y.dtype
        self.eval()
        self.pm.eval()

        # start at posterior mean
        x_t = self.pm(y.to(device)).to(dtype)  # no clamp here; clamp only for viz if your data are [0,1]
        dt  = 1.0 / float(steps)
        y_embed = y_embed_default(y.to(device))

        for i in range(steps):
            # midpoint time
            t_mid = torch.full((x_t.size(0), 1, 1, 1), (i + 0.5) * dt, device=device, dtype=dtype)

            # preimage under h_t^Y at current x_t (tied conjugacy)
            x_pre  = self.hflow.h_inv(x_t, t_mid, y_embed)
            v_base = self.rf(x_pre, t_mid)
            dt_h   = self.hflow.dt_h(x_pre, t_mid, y_embed)          # evaluate at preimage
            Dh_v   = self.hflow.jvp_h(x_pre, t_mid, y_embed, v_base) # evaluate at preimage
            v_tld  = dt_h + Dh_v


            x_t = x_t + v_tld * dt

        # (optional) clamp for display if your dynamic range is [0,1]
        return x_t
    
    def sample(self, y: torch.Tensor, steps: int = 50) -> torch.Tensor:
        if self.pmrf_only:
            return self.sample_pmrf(y, steps=steps)
        else:
            return self._sample_engrf(y, steps=steps)

    @torch.no_grad()
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _path_tuple(
        self, batch: Dict[str, torch.Tensor], *, resample_t: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x, y, x_star, delta, t, Z_t) with a single consistent t used for both Z_t and RF.
        """
        device = self._device()
        x = batch["x"].to(device)  # clean target (B,C,H,W)
        y = batch["y"].to(device)  # degraded/measurement
        B = x.size(0)
        # Always sample t here so every consumer uses the SAME t
        t = sample_t(B, device) if resample_t else batch.get("t", sample_t(B, device))

        with torch.no_grad():
            x_star = self.pm(y)             # posterior mean f_psi(y)

        if self.pmrf_sigma_s > 0:
            eps = torch.randn_like(x_star)
            Z0 = x_star + self.pmrf_sigma_s * eps
        else:
            Z0 = x_star

        delta = x - Z0    # Δ
        Z_t = (1.0 - t) * Z0 + t * x
        return x, y, x_star, delta, t, Z_t, Z0

    # ------------------------------ Stage 1 -------------------------------- #
    def compute_stage1(self, batch: Dict[str, torch.Tensor]):
        """
        Baseline FM on RF:
            Z_t = (1-t)X* + tX
            target = Δ = X - X*
            loss = E || v_theta(Z_t, t) - Δ ||^2
        """
        x, y, x_star, delta, t, Z_t,Z0 = self._path_tuple(batch, resample_t=True)

        # Always pass the SAME t that built Z_t
        v_pred = self.rf(Z_t, t)

        loss = fm_loss(v_pred, delta)
        return loss, {"fm_loss": float(loss.detach().cpu())}

    # ------------------------------ Stage 2 -------------------------------- #
    def compute_stage2(self, batch: Dict[str, torch.Tensor], lambda_conjugacy: float = 0.0):
        """
        Gauge-FM with tied conjugacy (endpoint-neutral):
            \tilde Z_t = h_t^Y(Z_t)
            target = ∂_t h_t^Y(Z_t) + (D h_t^Y)(Z_t) Δ
            pred   = ∂_t h_t^Y(\tilde Z_t) + (D h_t^Y)(\tilde Z_t) v_theta(h_t^{-1}(\tilde Z_t), t)

        Since we use the tied form, the conjugacy residual is identically zero; the optional
        'lambda_conjugacy' is provided for compatibility (will just add 0).
        """
        # IMPORTANT: do NOT resample t unless you also rebuild Z_t.
        x, y, x_star, delta, t, Z_t, Z0 = self._path_tuple(batch, resample_t=True)
        y_embed = y_embed_default(y)

        # Forward gauge path & gauged FM target (evaluated at Z_t)
        Z_t_tilde = self.hflow.h(Z_t, t, y_embed)            # h_t^Y(Z_t)
        dt_h_Zt   = self.hflow.dt_h(Z_t, t, y_embed)         # ∂_t h_t^Y(Z_t)
        Dh_delta  = self.hflow.jvp_h(Z_t, t, y_embed, delta) # (D h_t^Y)(Z_t) Δ
        target    = gfm_target(dt_h_Zt, Dh_delta)            # dt_h + Dh * Δ

        # Conjugated velocity prediction at the gauged point (evaluated at \tilde Z_t)
        Z_pre     = self.hflow.h_inv(Z_t_tilde, t, y_embed)   # ~ Z_t
        v_base    = self.rf(Z_pre, t)                        # v(Z_t, t)
        dt_h_Z    = self.hflow.dt_h(Z_pre, t, y_embed)       # ∂_t h_t(Z_t)
        Dh_v_Z    = self.hflow.jvp_h(Z_pre, t, y_embed, v_base)  # (D h_t)(Z_t) v(Z_t,t)
        v_tilde_pred = dt_h_Z + Dh_v_Z  

        loss_main = gfm_loss(v_tilde_pred, target)

        # In tied form r_t == 0, but keep API for future ablations
        if lambda_conjugacy > 0.0:
            # Recompute numerically to expose tiny integration/JVP mismatches:
            with torch.no_grad():
                Z_pre_dbg = self.hflow.h_inv(Z_t_tilde.detach(), t.detach(), y_embed.detach())  # ~ Z_t
                v_base_dbg = self.rf(Z_pre_dbg, t.detach()).detach()
                dt_h_dbg   = self.hflow.dt_h(Z_pre_dbg, t.detach(), y_embed.detach()).detach()
                Dh_v_dbg   = self.hflow.jvp_h(Z_pre_dbg, t.detach(), y_embed.detach(), v_base_dbg).detach()
                conj_rhs   = dt_h_dbg + Dh_v_dbg                        # RHS at Z_t
            resid = (v_tilde_pred - conj_rhs)
            loss_resid = (resid.square().mean())
            loss = loss_main + lambda_conjugacy * loss_resid
            logs = {"gfm_loss": float(loss_main.detach().cpu()),
                    "conj_resid": float(loss_resid.detach().cpu())}
        else:
            loss = loss_main
            logs = {"gfm_loss": float(loss_main.detach().cpu()),
                    "conj_resid": 0.0}

        return loss, logs