
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassifierConfig:
    # CIFAR-10 values from Table A.2 (Appendix).
    latent_dim: int = 256
    hidden_width: int = 512  # 2d for d=256
    num_classes: int = 10


class MLPClassifier(nn.Module):
    """MLP classifier θ in the latent space (Appendix B.2).
    Two hidden layers of width 2d with SiLU activations.
    """

    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        d = cfg.latent_dim
        w = cfg.hidden_width
        c = cfg.num_classes

        self.net = nn.Sequential(
            nn.Linear(d, w),
            nn.SiLU(),
            nn.Linear(w, w),
            nn.SiLU(),
            nn.Linear(w, c),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,d)
        return self.net(z)

    @staticmethod
    def cross_entropy_logits(logits: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return F.cross_entropy(logits, y, reduction=reduction)

    def l2_params(self) -> torch.Tensor:
        s = torch.zeros((), device=next(self.parameters()).device)
        for p in self.parameters():
            s = s + torch.sum(p ** 2)
        return s
