import torch, math
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Blocks ----------------------------- #

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.act(self.bn1(self.c1(x)))
        x = self.act(self.bn2(self.c2(x)))
        return x

class PatchProj(nn.Module):
    """1x1 projection to set attention channel dim without changing HxW."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1)
    def forward(self, x):
        return self.proj(x)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ch_half = int((channels + 1) // 2)
        inv = 1. / (10000 ** (torch.arange(0, ch_half, 2).float() / ch_half))
        self.register_buffer('inv', inv)
        self.ch_half = ch_half
        self.channels = channels
    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        device = x.device
        tH = torch.arange(H, device=device).type(self.inv.type())
        tW = torch.arange(W, device=device).type(self.inv.type())
        sinH = torch.einsum('i,j->ij', tH, self.inv)
        sinW = torch.einsum('i,j->ij', tW, self.inv)
        embH = torch.cat([sinH.sin(), sinH.cos()], dim=-1)   # H x ch_half
        embW = torch.cat([sinW.sin(), sinW.cos()], dim=-1)   # W x ch_half
        # build (H,W,C)
        E = torch.zeros(H, W, self.ch_half*2, device=device, dtype=x.dtype)
        E[:, :, :self.ch_half] = embW.unsqueeze(0).expand(H, -1, -1)
        E[:, :, self.ch_half:self.ch_half*2] = embH.unsqueeze(1).expand(-1, W, -1)
        E = E[..., :C]  # trim if needed
        return E.permute(2,0,1).unsqueeze(0).expand(B,-1,-1,-1)  # (B,C,H,W)

# ----------------------------- Attention ----------------------------- #

class TokenLinear(nn.Module):
    """Apply the same Linear to every token in a (B,N,D) tensor."""
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_out, d_in))
        self.b = nn.Parameter(torch.zeros(d_out)) if bias else None
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.b is not None:
            fan_in = d_in
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)
    def forward(self, x):          # (B,N,Din)
        y = x @ self.w.t()         # (B,N,Dout)
        return y + self.b if self.b is not None else y

class SelfAttention2D(nn.Module):
    """Single-head self-attention over HxW tokens with sinusoidal 2D PE."""
    def __init__(self, channels):
        super().__init__()
        self.to_q = TokenLinear(channels, channels)
        self.to_k = TokenLinear(channels, channels)
        self.to_v = TokenLinear(channels, channels)
        self.pe = PositionalEncodingPermute2D(channels)
        self.scale = channels ** -0.5
    def forward(self, x):  # (B,C,H,W)
        B,C,H,W = x.shape
        x = x + self.pe(x)
        X = x.flatten(2).transpose(1,2)             # (B,N,C), N=H*W
        Q,K,V = self.to_q(X), self.to_k(X), self.to_v(X)
        A = (Q @ K.transpose(1,2)) * self.scale     # (B,N,N)
        A = A.softmax(dim=-1)
        Y = A @ V                                    # (B,N,C)
        Y = Y.transpose(1,2).reshape(B,C,H,W)
        return Y

class CrossAttention2D(nn.Module):
    """
    Cross-attend decoder Y (query) to skip S (key,value). Operates at identical resolution.
    Channels are projected to a shared d.
    """
    def __init__(self, c_y, c_s, d=None):
        super().__init__()
        d = d or c_s
        self.q_proj = PatchProj(c_y, d)
        self.k_proj = PatchProj(c_s, d)
        self.v_proj = PatchProj(c_s, d)
        self.out_proj = PatchProj(d, c_s)  # back to skip channels
        self.pe_q = PositionalEncodingPermute2D(d)
        self.pe_k = PositionalEncodingPermute2D(d)
        self.scale = d ** -0.5
    def forward(self, Y, S):  # both (B,*,H,W) with same H,W
        assert Y.shape[-2:] == S.shape[-2:], "CrossAttention2D expects matched spatial size"
        B, _, H, W = Y.shape
        Q = self.q_proj(Y); K = self.k_proj(S); V = self.v_proj(S)
        Q = Q + self.pe_q(Q); K = K + self.pe_k(K)
        Q = Q.flatten(2).transpose(1,2)  # (B,N,d)
        K = K.flatten(2).transpose(1,2)
        V = V.flatten(2).transpose(1,2)
        A = (Q @ K.transpose(1,2)) * self.scale
        A = A.softmax(dim=-1)
        Z = A @ V                            # (B,N,d)
        Z = Z.transpose(1,2).reshape(B, -1, H, W)
        return self.out_proj(Z)              # (B,c_s,H,W)

# ----------------------------- Model ----------------------------- #

class UNetTransformerStrict(nn.Module):
    """
    Strict UNet with transformer:
      - Encoder/decoder like your UNet (MaxPool down, ConvTranspose2d up).
      - Bottleneck SelfAttention2D.
      - CrossAttention2D at each skip (decoder attends to encoder feature).
      - No interpolation; exact shape matches enforced.
    Requirement: H,W divisible by 2^(depth-1).
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, depth=4, attn_dim=None):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        chs = [base_ch * (2**i) for i in range(depth)]  # e.g. [64,128,256,512]
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(ConvBlock(prev, c))
            prev = c

        # Bottleneck
        self.mid_conv = ConvBlock(prev, prev)
        self.mid_attn = SelfAttention2D(prev)

        # Decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.xattn = nn.ModuleList()  # cross-attn blocks per stage

        # Stages: prev starts at chs[-1] (top), go down to chs[-2],...,chs[0]
        for c in reversed(chs[:-1]):
            # upsample channels prev -> c (spatial x2)
            self.up.append(nn.ConvTranspose2d(prev, c, kernel_size=2, stride=2))
            # cross-attention: query from upsampled (c), keys/values from skip (c)
            self.xattn.append(CrossAttention2D(c_y=c, c_s=c, d=attn_dim or c))
            # decoder conv after concatenating [cross_attended_skip(c), up_feat(c)] -> c
            self.dec.append(ConvBlock(in_ch=2*c, out_ch=c))
            prev = c

        self.out = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        down_factor = 2 ** (self.depth - 1)
        assert (H % down_factor == 0) and (W % down_factor == 0), \
            f"Input size {(H,W)} must be divisible by {down_factor}"

        # Encoder
        skips = []
        h = x
        for i, blk in enumerate(self.enc):
            h = blk(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.pool(h)

        # Bottleneck
        h = self.mid_conv(h)
        h = self.mid_attn(h)

        # Decoder
        for stage in range(self.depth - 1):
            h = self.up[stage](h)                          # (B,c,H',W')
            skip = skips[-(stage + 2)]                     # matching resolution
            assert h.shape[-2:] == skip.shape[-2:], \
                f"Upsample/skip mismatch: up={h.shape[-2:]}, skip={skip.shape[-2:]}"
            # cross-attend decoder query to skip; returns refined skip with same channels
            s_ref = self.xattn[stage](h, skip)             # (B,c,*,*)
            h = torch.cat([h, s_ref], dim=1)               # (B,2c,*,*)
            h = self.dec[stage](h)                         # (B,c,*,*)

        y = self.out(h)
        assert y.shape[-2:] == (H, W), "Output must match input spatial size"
        return y
