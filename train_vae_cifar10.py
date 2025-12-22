
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from wmg.data.cifar10 import CIFAR10DataConfig, make_cifar10_loaders
from wmg.losses.perceptual import LPIPSLoss
from wmg.models.vae import VAE, VAEConfig, kl_divergence
from wmg.utils.ema import EMA
from wmg.utils.misc import seed_everything, ensure_dir, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR-10 VAE (Table A.2).")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/cifar10_vae")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Table A.2 defaults
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--kl_weight", type=float, default=1e-2)
    p.add_argument("--perceptual_weight", type=float, default=0.2)
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--no_lpips", action="store_true", help="Disable LPIPS perceptual loss (deviates from paper).")

    # Weights & Biases logging
    p.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="worst-case-generation-MLP")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags.")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=50, help="Log scalar metrics every N steps.")
    p.add_argument("--wandb_log_images_every", type=int, default=10, help="Log recon/sample images every N epochs.")
    p.add_argument("--wandb_num_images", type=int, default=16, help="Number of images to log for recon/samples.")
    p.add_argument(
        "--wandb_log_ckpt_every",
        type=int,
        default=0,
        help="Upload checkpoints as W&B artifacts every N epochs (0 = only final).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)

    device = torch.device(args.device)

    data_cfg = CIFAR10DataConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = make_cifar10_loaders(data_cfg, train=True, augment=True, shuffle=True)

    vae_cfg = VAEConfig(latent_dim=256, hidden_channels=(128, 256, 512, 512), groups=32)
    vae = VAE(vae_cfg).to(device)

    lpips_loss: Optional[nn.Module] = None
    if (not args.no_lpips) and args.perceptual_weight > 0:
        lpips_loss = LPIPSLoss(net="vgg").to(device)

    opt = AdamW(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    sched = CosineAnnealingLR(opt, T_max=total_steps)

    ema = EMA(vae, decay=args.ema_decay)

    wandb_mod = None
    wandb_run = None
    use_wandb = bool(args.wandb) and args.wandb_mode != "disabled"
    if use_wandb:
        try:
            import wandb as wandb_mod  # type: ignore[assignment]
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "Failed to import wandb. Install it with `pip install wandb` or run without `--wandb`."
            ) from e

        tags = None
        if args.wandb_tags:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

        vae_cfg_dict = {
            "latent_dim": int(vae_cfg.latent_dim),
            "hidden_channels": list(vae_cfg.hidden_channels),
            "groups": int(vae_cfg.groups),
        }
        data_cfg_dict = {
            "data_root": str(data_cfg.data_root),
            "batch_size": int(data_cfg.batch_size),
            "num_workers": int(data_cfg.num_workers),
            "color_jitter": list(data_cfg.color_jitter),
            "normalize_to_minus1_1": bool(data_cfg.normalize_to_minus1_1),
        }
        run_name = args.wandb_name or f"cifar10_vae_seed{args.seed}"

        wandb_run = wandb_mod.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            tags=tags,
            mode=args.wandb_mode,
            dir=str(out_dir),
            config={
                **vars(args),
                "vae_cfg": vae_cfg_dict,
                "data_cfg": data_cfg_dict,
                "total_steps": int(total_steps),
                "train_batches_per_epoch": int(len(train_loader)),
                "train_dataset_size": int(len(train_loader.dataset)),
                "lpips_enabled": bool(lpips_loss is not None),
                "num_params": int(sum(p.numel() for p in vae.parameters())),
                "torch_version": torch.__version__,
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
            },
        )
        wandb_mod.define_metric("global_step")
        wandb_mod.define_metric("epoch")
        wandb_mod.define_metric("train/*", step_metric="global_step")
        wandb_mod.define_metric("epoch/*", step_metric="epoch")
        wandb_mod.define_metric("viz/*", step_metric="global_step")

    def _log_images(epoch: int, *, step: int, x_vis_cpu: Optional[torch.Tensor] = None) -> None:
        if wandb_mod is None or wandb_run is None:
            return
        if args.wandb_num_images <= 0:
            return
        if not ((epoch % args.wandb_log_images_every) == 0 or epoch in {1, args.epochs}):
            return

        from torchvision.utils import make_grid

        vae.eval()
        with torch.no_grad():
            if x_vis_cpu is None:
                x_vis_cpu, _ = next(iter(train_loader))
                x_vis_cpu = x_vis_cpu[: args.wandb_num_images].clone()

            x_vis = x_vis_cpu.to(device, non_blocking=True)
            num_vis = int(x_vis.shape[0])
            x_rec_vis, _mu, _logvar, _z = vae(x_vis)

            z = torch.randn(num_vis, 16, 4, 4, device=device)
            x_samp = vae.decode(z)

        def to_uint8_grid(imgs: torch.Tensor, *, nrow: int) -> "torch.Tensor":
            imgs = (imgs.detach().clamp(-1.0, 1.0) + 1.0) * 0.5  # [-1,1] -> [0,1]
            grid = make_grid(imgs, nrow=nrow, padding=2)
            return (grid.mul(255.0).round().to(torch.uint8)).cpu()

        # Reconstructions: top row = originals, bottom row = recon.
        recon_pair = torch.cat([x_vis, x_rec_vis], dim=0)
        recon_grid = to_uint8_grid(recon_pair, nrow=num_vis)
        recon_np = recon_grid.permute(1, 2, 0).numpy()

        nrow = int(math.sqrt(num_vis))
        nrow = max(1, min(num_vis, nrow))
        samp_grid = to_uint8_grid(x_samp, nrow=nrow)
        samp_np = samp_grid.permute(1, 2, 0).numpy()

        wandb_mod.log(
            {
                "viz/reconstructions": wandb_mod.Image(recon_np, caption=f"epoch {epoch} (top: x, bottom: recon)"),
                "viz/prior_samples": wandb_mod.Image(samp_np, caption=f"epoch {epoch} (z~N(0,I))"),
            },
            step=step,
        )
        vae.train()

    def _log_checkpoint(path: Path, *, epoch: int, step: int) -> None:
        if wandb_mod is None or wandb_run is None:
            return
        if args.wandb_log_ckpt_every <= 0 and epoch != args.epochs:
            return
        if epoch != args.epochs and (epoch % args.wandb_log_ckpt_every) != 0:
            return
        artifact = wandb_mod.Artifact(
            name="cifar10_vae",
            type="model",
            metadata={"epoch": int(epoch), "global_step": int(step)},
        )
        artifact.add_file(str(path))
        aliases = ["latest", f"epoch{epoch:04d}"]
        wandb_run.log_artifact(artifact, aliases=aliases)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        vae.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        l1_sum = 0.0
        kl_sum = 0.0
        perc_sum = 0.0
        num_batches = 0
        log_images_this_epoch = (
            wandb_mod is not None
            and args.wandb_num_images > 0
            and ((epoch % args.wandb_log_images_every) == 0 or epoch in {1, args.epochs})
        )
        x_vis_cpu: Optional[torch.Tensor] = None
        for x, _y in pbar:
            step_started = time.perf_counter()
            step = global_step + 1
            log_this_step = wandb_mod is not None and (step % max(1, args.wandb_log_every)) == 0

            if log_images_this_epoch and x_vis_cpu is None:
                x_vis_cpu = x[: args.wandb_num_images].clone()

            x = x.to(device, non_blocking=True)

            x_rec, mu, logvar, _z = vae(x)

            l1 = (x_rec - x).abs().mean()
            kl = kl_divergence(mu, logvar).mean()

            loss = l1 + args.kl_weight * kl
            perceptual = None
            if lpips_loss is not None:
                # LPIPS expects inputs roughly in [-1,1]
                perceptual = lpips_loss(x_rec, x)
                loss = loss + args.perceptual_weight * perceptual

            opt.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = None
            if log_this_step:
                gn = 0.0
                for p in vae.parameters():
                    if p.grad is not None:
                        gn += float(p.grad.detach().pow(2).sum().cpu())
                grad_norm = gn ** 0.5

            opt.step()
            ema.update()

            sched.step()
            global_step = step

            loss_f = float(loss.detach().cpu())
            l1_f = float(l1.detach().cpu())
            kl_f = float(kl.detach().cpu())
            perc_f = float(perceptual.detach().cpu()) if perceptual is not None else None

            loss_sum += loss_f
            l1_sum += l1_f
            kl_sum += kl_f
            if perc_f is not None:
                perc_sum += perc_f
            num_batches += 1

            pbar.set_postfix(loss=loss_f, l1=l1_f, kl=kl_f)

            if log_this_step:
                step_time_s = max(1e-12, time.perf_counter() - step_started)
                lr = float(opt.param_groups[0]["lr"])
                log_payload = {
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "train/loss": loss_f,
                    "train/l1": l1_f,
                    "train/kl": kl_f,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/step_time_s": step_time_s,
                    "train/images_per_s": float(x.shape[0]) / step_time_s,
                    "stats/mu_mean": float(mu.detach().mean().cpu()),
                    "stats/mu_std": float(mu.detach().std(unbiased=False).cpu()),
                    "stats/logvar_mean": float(logvar.detach().mean().cpu()),
                    "stats/logvar_std": float(logvar.detach().std(unbiased=False).cpu()),
                    "stats/x_rec_min": float(x_rec.detach().min().cpu()),
                    "stats/x_rec_max": float(x_rec.detach().max().cpu()),
                }
                if perc_f is not None:
                    log_payload["train/perceptual"] = perc_f
                if device.type == "cuda":
                    log_payload.update({
                        "gpu/mem_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
                        "gpu/mem_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
                    })
                wandb_mod.log(log_payload, step=global_step)

        if (epoch % args.save_every) == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "vae_cfg": vae_cfg.__dict__,
                "vae_state": vae.state_dict(),
                "ema_shadow": ema.shadow,
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(),
                "args": vars(args),
            }
            ckpt_path = out_dir / f"vae_epoch{epoch:04d}.pt"
            save_checkpoint(ckpt_path, ckpt)
            _log_checkpoint(ckpt_path, epoch=epoch, step=global_step)

        epoch_time_s = max(1e-12, time.perf_counter() - epoch_started)
        if wandb_mod is not None:
            denom = max(1, num_batches)
            wandb_mod.log(
                {
                    "epoch": int(epoch),
                    "epoch/avg_loss": float(loss_sum) / denom,
                    "epoch/avg_l1": float(l1_sum) / denom,
                    "epoch/avg_kl": float(kl_sum) / denom,
                    "epoch/avg_perceptual": (float(perc_sum) / denom) if lpips_loss is not None else 0.0,
                    "epoch/time_s": epoch_time_s,
                },
                step=global_step,
            )
            _log_images(epoch, step=global_step, x_vis_cpu=x_vis_cpu)

    # Save final
    final_path = out_dir / "vae_final.pt"
    save_checkpoint(final_path, {
        "epoch": args.epochs,
        "global_step": global_step,
        "vae_cfg": vae_cfg.__dict__,
        "vae_state": vae.state_dict(),
        "ema_shadow": ema.shadow,
        "args": vars(args),
    })
    _log_checkpoint(final_path, epoch=args.epochs, step=global_step)
    if wandb_mod is not None:
        wandb_mod.finish()


if __name__ == "__main__":
    main()
