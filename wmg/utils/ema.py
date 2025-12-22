
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class EMA:
    """Exponential moving average of model parameters."""

    model: nn.Module
    decay: float = 0.999

    def __post_init__(self):
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self):
        d = float(self.decay)
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, target_model: nn.Module):
        for name, p in target_model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])
