import torch, torch.nn as nn
from .unet import UNet
from  .swinir  import SwinIR

class PosteriorMean(nn.Module):
    def __init__(self, mode='unet', pm_cfg=None):
        super().__init__()
        self.mode = mode
        if mode == 'unet':
            self.net = UNet(**pm_cfg)
        elif mode == 'swinir':
            self.net = SwinIR(**pm_cfg)
        elif mode == 'identity':
            self.net = None
        else:
            raise ValueError(f'posterior_mean must be unet, swinir, or identity; got {mode}')

    def forward(self, y):
        return y if self.mode=='identity' else self.net(y)
        