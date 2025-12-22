
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """A simple ResNet block used in the VAE, matching Appendix B.4:
    - GroupNorm (32 groups) + SiLU
    - 3x3 conv
    - GroupNorm + SiLU
    - 3x3 conv
    - 1x1 shortcut when in/out channels differ
    """

    def __init__(self, in_channels: int, out_channels: int, *, groups: int = 32):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.skip is not None:
            x = self.skip(x)
        return x + h


class Downsample(nn.Module):
    """Average pooling downsample by factor 2."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample by factor 2."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="nearest")
