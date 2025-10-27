import torch, torch.nn as nn, torch.nn.functional as F

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

class UNet(nn.Module):
    """
    Strict UNet: no padding, no interpolation.
    Pools exactly (depth-1) times; upsamples exactly (depth-1) times.
    Requirement: H,W must be divisible by 2^(depth-1).
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, depth=4):
        super().__init__()
        assert depth >= 2, "depth >= 2 required"
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth)]  # e.g. [64,128,256,512]

        self.enc = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        prev = in_ch
        for c in chs:
            self.enc.append(ConvBlock(prev, c))
            prev = c

        self.mid = ConvBlock(prev, prev)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        # For depth=4 => up stages for 512->256->128->64 (3 stages)
        for c in reversed(chs[:-1]):  # 256, 128, 64
            self.up.append(nn.ConvTranspose2d(prev, c, kernel_size=2, stride=2))
            self.dec.append(ConvBlock(prev, c))  # prev = up(c) + skip(c) along channel dim later
            prev = c

        self.out = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Enforce clean divisibility to avoid any resizing
        down_factor = 2 ** (self.depth - 1)
        assert (H % down_factor == 0) and (W % down_factor == 0), \
            f"Input size {(H,W)} must be divisible by {down_factor} without padding/interp."

        feats = []
        h = x
        # Encoder with exactly (depth-1) pools
        for i, blk in enumerate(self.enc):
            h = blk(h)
            feats.append(h)
            if i < self.depth - 1:
                h = self.pool(h)

        # Bottleneck
        h = self.mid(h)

        # Decoder: transposed conv (×2) then concat with matching skip and decode
        # Walk stages: i=0 uses last skip feats[-1], i=1 uses feats[-2], etc.
        for stage in range(self.depth - 1):
            h = self.up[stage](h)
            skip = feats[-(stage + 2)]  # skip from encoder before corresponding pool
            # Strict size equality (no interpolate/pad)
            assert h.shape[-2:] == skip.shape[-2:], \
                f"Upsample/skip mismatch: up={h.shape[-2:]}, skip={skip.shape[-2:]}"
            h = torch.cat([h, skip], dim=1)  # channels: c + c = 2c, but ConvBlock expects 'prev'
            h = self.dec[stage](h)

        y = self.out(h)
        # Final strict check
        assert y.shape[-2:] == (H, W), f"Output size {y.shape[-2:]} != input size {(H, W)}"
        return y




