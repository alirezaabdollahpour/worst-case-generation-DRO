
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .resnet_blocks import ResBlock, Downsample, Upsample


@dataclass(frozen=True)
class VAEConfig:
    # CIFAR-10 values from Table A.2.
    latent_dim: int = 256  # d
    hidden_channels: Tuple[int, int, int, int] = (128, 256, 512, 512)
    groups: int = 32


class Encoder(nn.Module):
    """CIFAR-10 VAE encoder (Appendix B.4).
    Produces (mu, logvar) with latent code tensor shape (C=16, H=4, W=4) => d=256.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        assert cfg.latent_dim == 256, "This CIFAR-10 VAE assumes d=256 = 16*4*4."
        ch0, ch1, ch2, ch3 = cfg.hidden_channels

        self.conv_in = nn.Conv2d(3, ch0, kernel_size=3, stride=1, padding=1)

        # Resolution: 32x32
        self.rb_32_0 = ResBlock(ch0, ch0, groups=cfg.groups)
        self.rb_32_1 = ResBlock(ch0, ch1, groups=cfg.groups)  # 128 -> 256
        self.down_32 = Downsample()

        # Resolution: 16x16
        self.rb_16_0 = ResBlock(ch1, ch1, groups=cfg.groups)
        self.rb_16_1 = ResBlock(ch1, ch2, groups=cfg.groups)  # 256 -> 512
        self.down_16 = Downsample()

        # Resolution: 8x8
        self.rb_8_0 = ResBlock(ch2, ch2, groups=cfg.groups)
        self.rb_8_1 = ResBlock(ch2, ch3, groups=cfg.groups)  # 512 -> 512
        self.down_8 = Downsample()

        # Resolution: 4x4
        self.rb_4 = ResBlock(ch3, ch3, groups=cfg.groups)

        self.norm_out = nn.GroupNorm(num_groups=cfg.groups, num_channels=ch3, eps=1e-6, affine=True)
        # latent channels = 16, output mu and logvar => 32 channels
        self.conv_out = nn.Conv2d(ch3, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)

        h = self.rb_32_0(h)
        h = self.rb_32_1(h)
        h = self.down_32(h)

        h = self.rb_16_0(h)
        h = self.rb_16_1(h)
        h = self.down_16(h)

        h = self.rb_8_0(h)
        h = self.rb_8_1(h)
        h = self.down_8(h)

        h = self.rb_4(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        mu, logvar = torch.chunk(h, chunks=2, dim=1)  # each (B,16,4,4)
        return mu, logvar


class Decoder(nn.Module):
    """CIFAR-10 VAE decoder (Appendix B.4)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        assert cfg.latent_dim == 256, "This CIFAR-10 VAE assumes d=256 = 16*4*4."
        ch0, ch1, ch2, ch3 = cfg.hidden_channels

        self.conv_in = nn.Conv2d(16, ch3, kernel_size=1, stride=1, padding=0)

        # Resolution: 4x4
        self.rb_4 = ResBlock(ch3, ch3, groups=cfg.groups)

        # Upsample to 8x8, keep 512 channels (matches hidden_channels[2]/[3])
        self.up_4 = Upsample()
        self.rb_8_0 = ResBlock(ch3, ch3, groups=cfg.groups)
        self.rb_8_1 = ResBlock(ch3, ch3, groups=cfg.groups)

        # Upsample to 16x16, reduce to 256 channels (hidden_channels[1])
        self.up_8 = Upsample()
        self.rb_16_0 = ResBlock(ch3, ch1, groups=cfg.groups)  # 512 -> 256
        self.rb_16_1 = ResBlock(ch1, ch1, groups=cfg.groups)  # 256 -> 256

        # Upsample to 32x32, reduce to 128 channels (hidden_channels[0])
        self.up_16 = Upsample()
        self.rb_32_0 = ResBlock(ch1, ch0, groups=cfg.groups)  # 256 -> 128
        self.rb_32_1 = ResBlock(ch0, ch0, groups=cfg.groups)  # 128 -> 128

        self.norm_out = nn.GroupNorm(num_groups=cfg.groups, num_channels=ch0, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(ch0, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,16,4,4)
        h = self.conv_in(z)
        h = self.rb_4(h)

        h = self.up_4(h)
        h = self.rb_8_0(h)
        h = self.rb_8_1(h)

        h = self.up_8(h)
        h = self.rb_16_0(h)
        h = self.rb_16_1(h)

        h = self.up_16(h)
        h = self.rb_32_0(h)
        h = self.rb_32_1(h)

        h = F.silu(self.norm_out(h))
        x_rec = self.conv_out(h)
        return x_rec


class VAE(nn.Module):
    """Convolutional VAE used as the latent/concept space (Appendix B.4)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar, z


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)) for diagonal Gaussian posterior.
    Returns per-sample KL (shape: (B,)).
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=(1, 2, 3))
