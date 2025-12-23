
from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def infinite_loader(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    path = Path(path)
    try:
        # PyTorch 2.6 changed the default of `weights_only` to True. We keep the
        # default here to benefit from safer loading when possible.
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if ("Weights only load failed" not in msg) and ("weights_only" not in msg) and ("Unsupported global" not in msg):
            raise

        # If a checkpoint contains a small custom state object (e.g., BBArmijoState),
        # allowlist it for weights-only loading when supported.
        try:
            import torch.serialization as torch_serialization  # type: ignore

            safe_globals = getattr(torch_serialization, "safe_globals", None)
            if safe_globals is not None:
                from wmg.utils.BB_Armijo import BBArmijoState

                with safe_globals([BBArmijoState]):
                    return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            pass

        # Fall back to legacy pickle loading for compatibility. Only do this for trusted checkpoints.
        warnings.warn(
            "Falling back to torch.load(weights_only=False) for checkpoint compatibility. "
            "Only do this for trusted checkpoints.",
            RuntimeWarning,
        )
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
