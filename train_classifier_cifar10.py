
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.optim import Adam
from tqdm import tqdm

from wmg.data.cifar10 import CIFAR10DataConfig, make_cifar10_loaders
from wmg.models.vae import VAE, VAEConfig
from wmg.models.classifier import MLPClassifier, ClassifierConfig
from wmg.utils.misc import seed_everything, ensure_dir, load_checkpoint, save_checkpoint, infinite_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR-10 latent-space classifier θ0 (Table A.2).")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--vae_ckpt", type=str, required=True, help="Path to trained VAE checkpoint (.pt).")
    p.add_argument("--out_dir", type=str, default="./runs/cifar10_classifier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Table A.2 defaults
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)  # ω
    p.add_argument("--train_batches", type=int, default=20000)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=500)
    return p.parse_args()


def load_vae_from_ckpt(ckpt_path: str, device: torch.device) -> VAE:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = VAEConfig(**ckpt["vae_cfg"])
    vae = VAE(cfg).to(device)
    vae.load_state_dict(ckpt["vae_state"])
    # If EMA present, copy EMA weights into vae for encoding.
    if "ema_shadow" in ckpt and isinstance(ckpt["ema_shadow"], dict):
        from wmg.utils.ema import EMA
        ema = EMA(vae, decay=0.0)
        ema.shadow = ckpt["ema_shadow"]
        ema.copy_to(vae)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def eval_acc(vae: VAE, clf: MLPClassifier, loader, device: torch.device) -> float:
    clf.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mu, _logvar = vae.encode(x)
        z = mu.flatten(1)
        logits = clf(z)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    clf.train()
    return correct / max(1, total)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)

    vae = load_vae_from_ckpt(args.vae_ckpt, device)

    data_cfg = CIFAR10DataConfig(data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = make_cifar10_loaders(data_cfg, train=True, augment=False, shuffle=True)
    test_loader = make_cifar10_loaders(data_cfg, train=False, augment=False, shuffle=False)

    clf_cfg = ClassifierConfig(latent_dim=256, hidden_width=512, num_classes=10)
    clf = MLPClassifier(clf_cfg).to(device)

    opt = Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_iter = infinite_loader(train_loader)
    for step in tqdm(range(1, args.train_batches + 1), desc="train θ0"):
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            mu, _logvar = vae.encode(x)
            z = mu.flatten(1)

        logits = clf(z)
        loss = clf.cross_entropy_logits(logits, y, reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step % args.eval_every) == 0 or step == args.train_batches:
            acc = eval_acc(vae, clf, test_loader, device)
            save_checkpoint(out_dir / f"classifier_step{step:06d}.pt", {
                "step": step,
                "clf_cfg": clf_cfg.__dict__,
                "clf_state": clf.state_dict(),
                "args": vars(args),
                "test_acc": acc,
            })

    save_checkpoint(out_dir / "classifier_final.pt", {
        "step": args.train_batches,
        "clf_cfg": clf_cfg.__dict__,
        "clf_state": clf.state_dict(),
        "args": vars(args),
    })


if __name__ == "__main__":
    main()
