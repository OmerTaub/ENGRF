# models/rectified_flow.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from inspect import signature
from natten import NeighborhoodAttention2D

# -------------------------- utils -------------------------- #

def _normalize_t(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    elif t.dim() == 2:
        t = t.view(t.size(0), 1, 1, 1)
    elif t.dim() != 4:
        raise ValueError(f"t must be (B,), (B,1), or (B,1,1,1); got {tuple(t.shape)}")
    return t.to(device=like.device, dtype=like.dtype)

# ---------------- time embedding (DiT-style) ---------------- #

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_dim: int = 256, max_freq: float = 1000.0):
        super().__init__()
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even.")
        self.time_dim = time_dim
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0, torch.log(torch.tensor(max_freq)), time_dim // 2)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_flat = t.view(t.size(0), 1)  # (B,1)
        angles = 2.0 * torch.pi * t_flat * self.freqs.to(t.device, t.dtype)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, time_dim)

class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)  # (B, hidden)

# --------- AdaLN-Zero modulation per DiT block (channels-last) --------- #

class AdaLNMod(nn.Module):
    """
    Produces per-channel (scale, shift, gate) from a shared time feature.
    """
    def __init__(self, time_feat: int, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_feat, dim * 3))
        # zero-init so residuals start near identity (DiT trick)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.mlp(t_feat)  # (B, 3*dim)
        scale, shift, gate = m.chunk(3, dim=-1)
        return scale, shift, gate

# ----------------------- attention blocks ----------------------- #

# Try to use NATTEN (NeighborhoodAttention2D) for local attention.
# Falls back to full self-attention if NATTEN is not available.
try:
    from natten import NeighborhoodAttention2D  # pip install natten (see PMRF README)
    NAT_AVAILABLE = True
except Exception:
    NeighborhoodAttention2D = None
    NAT_AVAILABLE = False

class LocalAttention(nn.Module):
    """
    Local neighborhood attention (NATTEN) in NHWC format.
    Fallback: full MultiheadAttention on flattened sequence.
    """
    def __init__(self, dim: int, num_heads: int, kernel_size: int = 7, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size

        params = signature(NeighborhoodAttention2D).parameters
        if "dim" in params:
            self.attn = NeighborhoodAttention2D(dim=dim, num_heads=num_heads, kernel_size=kernel_size)
        else:
            # older natten used 'embed_dim'
            self.attn = NeighborhoodAttention2D(embed_dim=dim, num_heads=num_heads, kernel_size=kernel_size)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        if NAT_AVAILABLE:
            y = self.attn(x)  # (B, H, W, C)
            return self.proj_drop(y)
        else:
            B, H, W, C = x.shape
            seq = x.view(B, H * W, C)          # (B, N, C)
            y, _ = self.attn(seq, seq, seq)    # full attention fallback
            y = y.view(B, H, W, C)
            return self.proj_drop(y)

class GlobalAttention(nn.Module):
    """
    Full self-attention at the bottleneck (global context).
    Uses efficient nn.MultiheadAttention on flattened sequence (NHWC in, NHWC out).
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        seq = x.view(B, H * W, C)
        y, _ = self.attn(seq, seq, seq)
        y = y.view(B, H, W, C)
        return self.proj_drop(y)

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.drop = nn.Dropout(drop)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)  # DiT-style zero init

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    """
    Generic (local/global) transformer block with AdaLN-Zero conditioning on time.
    Operates in NHWC. attention_fn must accept NHWC and return NHWC.
    """
    def __init__(self, dim: int, num_heads: int, time_feat: int, attention_fn: nn.Module, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.mod1 = AdaLNMod(time_feat, dim)
        self.mod2 = AdaLNMod(time_feat, dim)
        self.attn = attention_fn
        self.mlp = MLP(dim, hidden_mult=mlp_ratio)

    def forward(self, x: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,C); t_feat: (B,T)
        s1, b1, g1 = self.mod1(t_feat)                 # (B,C)
        h = self.ln1(x)
        h = h * (1 + s1.unsqueeze(1).unsqueeze(1)) + b1.unsqueeze(1).unsqueeze(1)
        x = x + g1.unsqueeze(1).unsqueeze(1) * self.attn(h)

        s2, b2, g2 = self.mod2(t_feat)
        h = self.ln2(x)
        h = h * (1 + s2.unsqueeze(1).unsqueeze(1)) + b2.unsqueeze(1).unsqueeze(1)
        x = x + g2.unsqueeze(1).unsqueeze(1) * self.mlp(h)
        return x

# ------------------------ patch & pyramid ops ------------------------ #

class PatchEmbed(nn.Module):
    """
    Conv patchify: (B,C,H,W) -> (B,H/P,W/P,C_embed) [NHWC]
    """
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                      # (B, embed, H/P, W/P)
        return x.permute(0, 2, 3, 1).contiguous()  # NHWC

class PatchUnembed(nn.Module):
    """
    ConvTranspose unpatchify: (B,H/P,W/P,C_embed) -> (B,out_ch,H,W)
    """
    def __init__(self, embed_dim: int, out_ch: int, patch_size: int):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch_size, stride=patch_size)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()  # NCHW
        return self.proj(x)

class Downsample(nn.Module):
    """ 2× downsample tokens via strided conv; doubles channels. """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,3,1,2).contiguous()
        x = self.conv(x)
        return x.permute(0,2,3,1).contiguous()

class Upsample(nn.Module):
    """ 2× upsample tokens via transposed conv; halves channels. """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,3,1,2).contiguous()
        x = self.deconv(x)
        return x.permute(0,2,3,1).contiguous()

# ----------------------- HDiT rectified flow ----------------------- #

class HDiTRF(nn.Module):
    """
    Hourglass Diffusion Transformer for rectified flow velocity prediction.
    Defaults mirror PMRF controlled setting (Table 12): levels 1+1, depths (2, 11),
    widths (384 -> 768), head_dim=64, NA kernel=7, patch=4. (Works for 208x208.)
    """
    def __init__(
        self,
        in_ch: int = 1,
        patch_size: int = 4,
        # local levels
        local_widths: List[int] = (384,),
        local_depths: List[int] = (2,),
        # global bottleneck
        global_width: int = 768,
        global_depth: int = 11,
        head_dim: int = 64,
        natten_kernel: int = 7,
        time_dim: int = 256,
        time_mlp_hidden: int = 1024,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        assert len(local_widths) == len(local_depths)
        self.patch = patch_size

        # time conditioning (shared across blocks)
        self.time_embed = SinusoidalTimeEmbedding(time_dim=time_dim)
        self.time_mlp = TimeMLP(time_dim, time_mlp_hidden)

        # --- encoder: patchify + one local stage (more stages supported via lists) ---
        self.patch_in = PatchEmbed(in_ch, local_widths[0], patch_size)

        self.local_stages_enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        dims = list(local_widths)
        for i, (dim, depth) in enumerate(zip(local_widths, local_depths)):
            heads = max(1, dim // head_dim)
            blocks = nn.ModuleList([
                TransformerBlock(dim, heads, time_mlp_hidden,
                                 LocalAttention(dim, heads, natten_kernel),
                                 mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ])
            self.local_stages_enc.append(blocks)
            # If there is another level OR the global stage expects a different width, downsample next
            next_dim = (local_widths[i+1] if i+1 < len(local_widths) else global_width)
            self.downs.append(Downsample(dim, next_dim))

        # --- bottleneck: global attention blocks at lowest token resolution ---
        g_heads = max(1, global_width // head_dim)
        self.global_proj_in = nn.Identity()  # already projected by last Downsample
        self.global_blocks = nn.ModuleList([
            TransformerBlock(global_width, g_heads, time_mlp_hidden,
                             GlobalAttention(global_width, g_heads),
                             mlp_ratio=mlp_ratio)
            for _ in range(global_depth)
        ])

        # --- decoder: upsample back with one local stage mirrored ---
        self.ups = nn.ModuleList()
        self.local_stages_dec = nn.ModuleList()
        for i in reversed(range(len(local_widths))):
            in_dim = (global_width if i == len(local_widths)-1 else local_widths[i+1])
            out_dim = local_widths[i]
            self.ups.append(Upsample(in_dim, out_dim))
            heads = max(1, out_dim // head_dim)
            # fuse + refine; one extra local block is typically enough
            self.local_stages_dec.append(nn.ModuleList([
                TransformerBlock(out_dim, heads, time_mlp_hidden,
                                 LocalAttention(out_dim, heads, natten_kernel),
                                 mlp_ratio=mlp_ratio)
                for _ in range(1)  # light refine after skip-add
            ]))

        self.out_norm = nn.LayerNorm(local_widths[0], elementwise_affine=True)
        self.patch_out = PatchUnembed(local_widths[0], in_ch, patch_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # time features
        t = _normalize_t(t, like=x_t)
        t_feat = self.time_mlp(self.time_embed(t))  # (B, Tfeat)

        # encoder local stage(s)
        x = self.patch_in(x_t)  # (B, H/p, W/p, C0)
        skips = []
        for blocks, down in zip(self.local_stages_enc, self.downs):
            for blk in blocks:
                x = blk(x, t_feat)
            skips.append(x)        # save for skip (same spatial size)
            x = down(x)            # 2x downsample + widen channels

        # global bottleneck
        for blk in self.global_blocks:
            x = blk(x, t_feat)

        # decoder local stage(s)
        for up, blocks, skip in zip(self.ups, self.local_stages_dec, reversed(skips)):
            x = up(x)
            # add skip (channel dims were matched by Upsample)
            if x.shape != skip.shape:
                # safe resize if needed (shouldn't trigger for divisible sizes)
                x = F.interpolate(x.permute(0,3,1,2), size=skip.shape[1:3], mode="nearest").permute(0,2,3,1)
            x = x + skip
            for blk in blocks:
                x = blk(x, t_feat)

        # project back to pixels
        x = self.out_norm(x)
        v = self.patch_out(x)
        return v

# ---------------------------- ImageTransformerV2 Wrapper ---------------------------- #

class ImageTransformerV2RF(nn.Module):
    """
    Wrapper for ImageTransformerDenoiserModelV2 to work with rectified flow interface.
    Adapts the k-diffusion style denoiser to accept (x_t, t) instead of (x, sigma).
    """
    def __init__(self, **cfg):
        super().__init__()
        from .image_transformer_v2 import (
            ImageTransformerDenoiserModelV2,
            LevelSpec,
            MappingSpec,
            GlobalAttentionSpec,
            NeighborhoodAttentionSpec,
            ShiftedWindowAttentionSpec,
            NoAttentionSpec,
        )
        
        # Extract configuration
        in_channels = cfg.get("in_channels", 1)
        out_channels = cfg.get("out_channels", in_channels)
        patch_size = cfg.get("patch_size", (2, 2))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        # Mapping network config
        mapping_cfg = cfg.get("mapping", {})
        mapping = MappingSpec(
            depth=mapping_cfg.get("depth", 2),
            width=mapping_cfg.get("width", 256),
            d_ff=mapping_cfg.get("d_ff", 512),
            dropout=mapping_cfg.get("dropout", 0.0),
        )
        
        # Level specs
        levels_cfg = cfg.get("levels", [])
        if not levels_cfg:
            # Default: single level with neighborhood attention
            levels_cfg = [{
                "depth": 4,
                "width": 256,
                "d_ff": 1024,
                "self_attn_type": "neighborhood",
                "d_head": 64,
                "kernel_size": 7,
                "dropout": 0.0,
            }]
        
        levels = []
        for level_cfg in levels_cfg:
            attn_type = level_cfg.get("self_attn_type", "neighborhood")
            if attn_type == "global":
                self_attn = GlobalAttentionSpec(d_head=level_cfg.get("d_head", 64))
            elif attn_type == "neighborhood":
                self_attn = NeighborhoodAttentionSpec(
                    d_head=level_cfg.get("d_head", 64),
                    kernel_size=level_cfg.get("kernel_size", 7),
                )
            elif attn_type == "shifted_window":
                self_attn = ShiftedWindowAttentionSpec(
                    d_head=level_cfg.get("d_head", 64),
                    window_size=level_cfg.get("window_size", 8),
                )
            elif attn_type == "no_attention" or attn_type == "none":
                self_attn = NoAttentionSpec()
            else:
                raise ValueError(f"Unknown attention type: {attn_type}. Must be one of: global, neighborhood, shifted_window, no_attention")
            
            levels.append(LevelSpec(
                depth=level_cfg.get("depth", 4),
                width=level_cfg.get("width", 256),
                d_ff=level_cfg.get("d_ff", 1024),
                self_attn=self_attn,
                dropout=level_cfg.get("dropout", 0.0),
            ))
        
        # Create the model
        self.model = ImageTransformerDenoiserModelV2(
            levels=levels,
            mapping=mapping,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            num_classes=cfg.get("num_classes", 0),
            mapping_cond_dim=cfg.get("mapping_cond_dim", 0),
            degradation_params_dim=cfg.get("degradation_params_dim", None),
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for rectified flow.
        
        Args:
            x_t: Input tensor at time t, shape (B, C, H, W)
            t: Time parameter, shape (B, 1, 1, 1) or (B,)
        
        Returns:
            Velocity prediction v(x_t, t)
        """
        # Flatten t to (B,) for sigma input
        if t.dim() > 1:
            t = t.view(t.size(0))
        
        # The model expects sigma; we use t directly as the time conditioning
        return self.model(x_t, sigma=t)


# ---------------------------- Public wrappers ---------------------------- #

class RectifiedFlow(nn.Module):
    """
    Wrapper that instantiates UNetRF, HDiTRF, or ImageTransformerV2RF via config.
    """
    def __init__(self, **rf_cfg: Dict):
        super().__init__()
        arch = rf_cfg.get("arch", "hdit")
        
        if arch == "unet":  # legacy UNet
            self.net = UNetRF(
                in_ch=rf_cfg.get("in_channels", 1),
                base_ch=rf_cfg.get("base_channels", 64),
                time_dim=rf_cfg.get("time_dim", 256),
                time_mlp_hidden=rf_cfg.get("time_mlp_hidden", 512),
                num_levels=rf_cfg.get("num_levels", 3),
                groups=rf_cfg.get("groups", 8),
            )
        elif arch == "hourglass" or arch == "image_transformer_v2":
            # k-diffusion style hourglass transformer
            self.net = ImageTransformerV2RF(**rf_cfg)
        else:
            # HDiT defaults per PMRF controlled-setup (Table 12). Works for 208×208.
            self.net = HDiTRF(
                in_ch=rf_cfg.get("in_channels", 1),
                patch_size=rf_cfg.get("patch_size", 4),
                local_widths=tuple(rf_cfg.get("local_widths", (384,))),
                local_depths=tuple(rf_cfg.get("local_depths", (2,))),
                global_width=rf_cfg.get("global_width", 768),
                global_depth=rf_cfg.get("global_depth", 11),
                head_dim=rf_cfg.get("head_dim", 64),
                natten_kernel=rf_cfg.get("natten_kernel", 7),
                time_dim=rf_cfg.get("time_dim", 256),
                time_mlp_hidden=rf_cfg.get("time_mlp_hidden", 1024),
                mlp_ratio=rf_cfg.get("mlp_ratio", 4),
            )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x_t, t)

    @torch.no_grad()
    def sample_euler(self, x0: torch.Tensor, steps: int = 50) -> torch.Tensor:
        dt = 1.0 / steps
        x_t = x0.clone()
        for i in range(steps):
            t = torch.full((x_t.size(0), 1, 1, 1), (i + 0.5) * dt, device=x_t.device, dtype=x_t.dtype)
            v = self.forward(x_t, t)
            x_t = x_t + v * dt
        return x_t

# ----------------------- legacy UNet (kept for fallback) ----------------------- #

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_feat: int, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(time_feat, 2 * out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        scale_shift = self.emb(t_feat).view(t_feat.size(0), 2 * self.out_ch, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.conv1(self.act(self.norm1(x)))
        h = h * (1 + scale) + shift
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x): return self.op(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x): return self.op(x)

class UNetRF(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 64, time_dim: int = 256, time_mlp_hidden: int = 512, num_levels: int = 3, groups: int = 8):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim=time_dim)
        self.time_mlp = TimeMLP(time_dim, time_mlp_hidden)
        widths = [base_ch * (2 ** i) for i in range(num_levels)]
        self.enc_in = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(num_levels):
            self.enc_blocks.append(ResBlock(widths[i], widths[i], time_mlp_hidden, groups))
            if i < num_levels - 1:
                self.downs.append(Down(widths[i], widths[i+1]))
        self.mid1 = ResBlock(widths[-1], widths[-1], time_mlp_hidden, groups)
        self.mid2 = ResBlock(widths[-1], widths[-1], time_mlp_hidden, groups)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(num_levels - 1)):
            self.ups.append(Up(widths[i+1], widths[i]))
            self.dec_blocks.append(ResBlock(2 * widths[i], widths[i], time_mlp_hidden, groups))
        self.out_norm = nn.GroupNorm(groups, widths[0])
        self.out = nn.Conv2d(widths[0], in_ch, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = _normalize_t(t, like=x_t)
        t_feat = self.time_mlp(self.time_embed(t))
        skips = []
        h = self.enc_in(x_t)
        for i, block in enumerate(self.enc_blocks):
            h = block(h, t_feat)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        h = self.mid1(h, t_feat)
        h = self.mid2(h, t_feat)
        for up, block in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop(-2)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_feat)
        h = F.silu(self.out_norm(h))
        v = self.out(h)
        return v
