
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss wrapper.

    Paper uses LPIPS with weight 0.2 for CIFAR-10 VAE training (Table A.2).
    This module tries to import the `lpips` package. If unavailable, it raises
    an informative error.

    Expected input range: [-1, 1], shape (B,3,H,W).
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        try:
            import lpips  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LPIPS package not found. Install with `pip install lpips` "
                "or set perceptual_weight=0."
            ) from e
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # lpips returns shape (B,1,1,1) or (B,1); reduce to scalar
        d = self.lpips(x, y)
        return d.mean()
