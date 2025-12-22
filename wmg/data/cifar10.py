
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CIFAR10DataConfig:
    data_root: str = "./data"
    batch_size: int = 250
    num_workers: int = 4
    # Augmentations used for VAE training per Table A.2: horizontal flip + color jitter.
    color_jitter: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.1)
    normalize_to_minus1_1: bool = True


def _normalize_transform(normalize_to_minus1_1: bool) -> transforms.Compose:
    if normalize_to_minus1_1:
        # CIFAR-10 images are in [0,1] after ToTensor(). Map to [-1,1].
        return transforms.Lambda(lambda t: t * 2.0 - 1.0)
    return transforms.Lambda(lambda t: t)


def make_cifar10_loaders(
    cfg: CIFAR10DataConfig,
    *,
    train: bool = True,
    augment: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    tfs = []
    if augment:
        tfs += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(*cfg.color_jitter),
        ]
    tfs += [
        transforms.ToTensor(),
        _normalize_transform(cfg.normalize_to_minus1_1),
    ]
    ds = datasets.CIFAR10(root=cfg.data_root, train=train, download=True, transform=transforms.Compose(tfs))
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=train,  # for consistent batch sizes in training loops
    )
    return loader


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
