
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from wmg.data.cifar10 import CIFAR10_CLASS_NAMES
from wmg.data.cifar10 import cifar10_transform
from wmg.models.vae import VAE, VAEConfig
from wmg.models.classifier import MLPClassifier, ClassifierConfig
from wmg.models.transport import LabelConditionedTransport, TransportConfig
from wmg.utils.misc import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Figure 4(b)-style interpolation panel on CIFAR-10.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--minimax_ckpt", type=str, required=True, help="Checkpoint with transport_state and theta0_state.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--steps", type=int, default=8, help="Interpolation steps (including endpoints).")
    p.add_argument("--out_path", type=str, default="./fig4_cifar10.png")
    return p.parse_args()


def _normalize_to_minus1_1() -> transforms.Compose:
    # Keep a single canonical normalization across all scripts.
    return cifar10_transform(normalize_to_minus1_1=True, augment=False)


def _assert_minus1_1(x: torch.Tensor, *, name: str, tol: float = 1e-3) -> None:
    x_min = float(x.detach().min().cpu())
    x_max = float(x.detach().max().cpu())
    if (x_min < -1.0 - tol) or (x_max > 1.0 + tol):
        raise ValueError(
            f"{name} expected in [-1,1] (got min={x_min:.4f}, max={x_max:.4f}). "
            "This script assumes CIFAR-10 tensors are mapped from [0,1] to [-1,1] before VAE encoding."
        )


def load_vae(ckpt_path: str, device: torch.device) -> VAE:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = VAEConfig(**ckpt["vae_cfg"])
    vae = VAE(cfg).to(device)
    vae.load_state_dict(ckpt["vae_state"])
    if "ema_shadow" in ckpt and isinstance(ckpt["ema_shadow"], dict):
        from wmg.utils.ema import EMA
        ema = EMA(vae, decay=0.0)
        ema.shadow = ckpt["ema_shadow"]
        ema.copy_to(vae)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def load_theta0_and_transport(minimax_ckpt_path: str, device: torch.device) -> Tuple[MLPClassifier, torch.nn.Module]:
    ckpt = load_checkpoint(minimax_ckpt_path, map_location=device)

    clf_cfg = ClassifierConfig(**ckpt.get("classifier_cfg", {"latent_dim": 256, "hidden_width": 512, "num_classes": 10}))
    theta0 = MLPClassifier(clf_cfg).to(device)
    theta0.load_state_dict(ckpt["theta0_state"])
    theta0.eval()
    for p in theta0.parameters():
        p.requires_grad_(False)

    tcfg_raw = ckpt.get("transport_cfg", {})
    ttype = ckpt.get("transport_type", None)

    # Backward/forward compatible transport selection:
    # - New checkpoints store `transport_type` = {"mlp","icnn"}.
    # - Older checkpoints only have the MLP config.
    # - Some experimental checkpoints may include a `type` key in transport_cfg.
    if ttype is None and isinstance(tcfg_raw, dict):
        ttype = tcfg_raw.get("type", None)
    if ttype is None:
        ttype = "icnn" if (isinstance(tcfg_raw, dict) and ("hidden_sizes" in tcfg_raw)) else "mlp"

    if isinstance(tcfg_raw, dict) and "type" in tcfg_raw:
        tcfg_raw = {k: v for k, v in tcfg_raw.items() if k != "type"}

    if ttype == "mlp":
        tcfg = TransportConfig(**tcfg_raw)
        transport = LabelConditionedTransport(tcfg).to(device)
    elif ttype == "icnn":
        from wmg.models.icnn import LabelConditionedICNNTransport, ICNNTransportConfig

        if isinstance(tcfg_raw, dict) and ("hidden_sizes" in tcfg_raw):
            hs = tcfg_raw["hidden_sizes"]
            if isinstance(hs, list):
                tcfg_raw = {**tcfg_raw, "hidden_sizes": tuple(hs)}
        icfg = ICNNTransportConfig(**tcfg_raw)
        transport = LabelConditionedICNNTransport(icfg).to(device)
    else:
        raise ValueError(f"Unknown transport_type='{ttype}' in checkpoint.")

    transport.load_state_dict(ckpt["transport_state"])
    transport.eval()
    for p in transport.parameters():
        p.requires_grad_(False)

    return theta0, transport


@torch.no_grad()
def decode_latents_to_images(vae: VAE, z_flat: torch.Tensor) -> torch.Tensor:
    # z_flat: (B,256)
    z = z_flat.view(-1, 16, 4, 4)
    x_rec = vae.decode(z)
    # Clamp to [-1,1] then map to [0,1] for visualization
    x_vis = torch.clamp(x_rec, -1.0, 1.0)
    x_vis = (x_vis + 1.0) * 0.5
    return x_vis


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    vae = load_vae(args.vae_ckpt, device)
    theta0, transport = load_theta0_and_transport(args.minimax_ckpt, device)

    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=_normalize_to_minus1_1())
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Collect latents and transported latents across test set, then pick top examples by θ0 loss on transported points.
    z_all = []
    zt_all = []
    y_all = []
    loss_all = []

    for x, y in tqdm(test_loader, desc="encode test"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _assert_minus1_1(x, name="CIFAR-10 test batch")

        with torch.no_grad():
            mu, _logvar = vae.encode(x)
            if not torch.isfinite(mu).all():
                raise RuntimeError("Non-finite VAE encoder output (mu). Check input normalization and checkpoint.")
            z = mu.flatten(1)

        zt = transport(z, y)
        if not torch.isfinite(zt).all():
            raise RuntimeError("Non-finite transport output (T(z,y)). Check transport checkpoint and inputs.")

        with torch.no_grad():
            logits_t = theta0(zt)
            loss_t = F.cross_entropy(logits_t, y, reduction="none")  # (B,)

        z_all.append(z.cpu())
        zt_all.append(zt.cpu())
        y_all.append(y.cpu())
        loss_all.append(loss_t.cpu())

    z_all = torch.cat(z_all, dim=0)
    zt_all = torch.cat(zt_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    loss_all = torch.cat(loss_all, dim=0)

    # Select top rows examples by transported loss.
    top_idx = torch.topk(loss_all, k=args.rows).indices

    steps = torch.linspace(0.0, 1.0, args.steps)
    fig, axes = plt.subplots(args.rows, args.steps, figsize=(args.steps * 2.0, args.rows * 2.0))

    if args.rows == 1:
        axes = axes[None, :]  # (1,steps)

    for r, idx in enumerate(top_idx.tolist()):
        z0 = z_all[idx].to(device)
        z1 = zt_all[idx].to(device)
        y_true = int(y_all[idx].item())

        # Predictions by θ0 at endpoints (as in Figure 4 caption).
        with torch.no_grad():
            p0 = int(theta0(z0[None, :].to(device)).argmax(dim=1).item())
            p1 = int(theta0(z1[None, :].to(device)).argmax(dim=1).item())

        # Interpolate in latent space and decode.
        z_interp = torch.stack([(1.0 - t) * z0 + t * z1 for t in steps], dim=0)  # (steps,256)
        x_interp = decode_latents_to_images(vae, z_interp).cpu()  # (steps,3,32,32)

        for c in range(args.steps):
            ax = axes[r, c]
            img = x_interp[c].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        # Annotate endpoint predicted labels; color red when incorrect.
        start_color = "red" if p0 != y_true else "black"
        end_color = "red" if p1 != y_true else "black"
        axes[r, 0].set_title(CIFAR10_CLASS_NAMES[p0], color=start_color, fontsize=10)
        axes[r, args.steps - 1].set_title(CIFAR10_CLASS_NAMES[p1], color=end_color, fontsize=10)

    plt.tight_layout()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
