
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransportConfig:
    # CIFAR-10 values from Table A.2 (Appendix).
    latent_dim: int = 256
    hidden_width: int = 512  # 2d for d=256
    num_classes: int = 10
    label_embed_dim: int = 512  # embedding size y


class LabelConditionedTransport(nn.Module):
    """Neural transport map T_φ(x, y) in the latent space (Appendix B.2).

    Implements identity initialization via a residual form:
        T_φ(x, y) = x + R_φ([x; emb(y)])
    and sets the final layer weights/bias to zero so R_φ ≡ 0 at init.
    """

    def __init__(self, cfg: TransportConfig):
        super().__init__()
        d = cfg.latent_dim
        w = cfg.hidden_width
        e = cfg.label_embed_dim

        self.embed = nn.Embedding(cfg.num_classes, e)
        self.mlp = nn.Sequential(
            nn.Linear(d + e, w),
            nn.SiLU(),
            nn.Linear(w, w),
            nn.SiLU(),
            nn.Linear(w, d),
        )

        # Zero-init final layer for identity map initialization
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B,d), y: (B,)
        ey = self.embed(y)
        h = torch.cat([x, ey], dim=1)
        r = self.mlp(h)
        return x + r
