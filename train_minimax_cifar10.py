
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from wmg.models.vae import VAE, VAEConfig
from wmg.models.classifier import MLPClassifier, ClassifierConfig
from wmg.models.transport import LabelConditionedTransport, TransportConfig
from wmg.utils.misc import seed_everything, ensure_dir, load_checkpoint, save_checkpoint, infinite_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimax training (GDA + matching) on CIFAR-10 latent space.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--clf_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./runs/cifar10_minimax")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Minimax / GDA hyperparameters (Figure 4 uses these for CIFAR-10)
    p.add_argument("--n_per_class", type=int, default=200)        # n = 200 per class => 2000
    p.add_argument("--k_batches", type=int, default=20000)        # k = 20,000
    p.add_argument("--batch_size", type=int, default=500)         # m = 500
    p.add_argument("--match_batch_size", type=int, default=100)   # m' = 100
    p.add_argument("--eta", type=float, default=1e-3)             # particle ascent step size η
    p.add_argument("--tau", type=float, default=1e-3)             # theta descent step size τ
    p.add_argument("--momentum", type=float, default=0.9)         # momentum ν_m

    # Objective hyperparameters
    p.add_argument("--gamma", type=float, default=8.0,
                  help="Quadratic penalty parameter γ (paper does not state CIFAR-10 value).")
    p.add_argument("--omega", type=float, default=1e-3,
                  help="Weight decay ω inside ℓ_y(θ, x). Table A.2 uses 1e-3.")

    # Transport optimizer (Table A.2)
    p.add_argument(
        "--transport",
        type=str,
        default="mlp",
        choices=["mlp", "icnn"],
        help="Transport map parametrization: residual MLP (Appendix B.2) or ICNN potential (wmg/models/icnn.py).",
    )
    p.add_argument("--transport_lr", type=float, default=1e-4)
    p.add_argument("--transport_wd", type=float, default=1e-5)

    # ICNN transport hyperparameters / BB+Armijo step-size
    p.add_argument("--icnn_hidden_sizes", type=str, default="512,512,512",
                   help="Comma-separated ICNN hidden widths (e.g. '512,512,512').")
    p.add_argument("--icnn_activation", type=str, default="softplus", choices=["softplus", "relu"])
    p.add_argument("--icnn_strong_convexity", type=float, default=1.0)
    p.add_argument("--icnn_nonneg_init", type=str, default="principled", choices=["principled", "xavier"])
    p.add_argument("--icnn_softplus_beta", type=float, default=20.0)
    p.add_argument("--icnn_no_identity_init", action="store_true",
                   help="Disable identity-like initialization for ICNN transport.")
    p.add_argument(
        "--icnn_bb_armijo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use BB+Armijo line search for ICNN transport parameter updates. "
            "If disabled, trains ICNN transport with Adam using --transport_lr/--transport_wd (same as MLP)."
        ),
    )
    p.add_argument("--icnn_alpha0", type=float, default=1e-1)
    p.add_argument("--icnn_alpha_min", type=float, default=1e-6)
    p.add_argument("--icnn_alpha_max", type=float, default=10.0)
    p.add_argument("--icnn_ls_c", type=float, default=1e-4)
    p.add_argument("--icnn_ls_shrink", type=float, default=0.5)
    p.add_argument("--icnn_ls_max_steps", type=int, default=10)

    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--metrics_every", type=int, default=50, help="Write metrics JSON every N steps.")
    p.add_argument("--metrics_file", type=str, default="metrics.json", help="Metrics JSON filename inside out_dir.")
    return p.parse_args()


def _normalize_to_minus1_1() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])


def load_vae(ckpt_path: str, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
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
    return vae, ckpt


def load_classifier(ckpt_path: str, device: torch.device) -> Tuple[MLPClassifier, Dict[str, Any]]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ClassifierConfig(**ckpt["clf_cfg"])
    clf = MLPClassifier(cfg).to(device)
    clf.load_state_dict(ckpt["clf_state"])
    clf.train()
    return clf, ckpt


@torch.no_grad()
def encode_subset_to_latents(
    vae: VAE,
    subset_loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    latents: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for x, y in subset_loader:
        x = x.to(device, non_blocking=True)
        mu, _logvar = vae.encode(x)       # (B,16,4,4)
        z = mu.flatten(1)                # (B,256)
        latents.append(z.cpu())
        labels.append(y.cpu())
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


def make_class_balanced_indices(targets: List[int], n_per_class: int, seed: int) -> List[int]:
    g = torch.Generator()
    g.manual_seed(seed)

    targets_t = torch.tensor(targets, dtype=torch.long)
    out: List[int] = []
    for c in range(10):
        idx = torch.nonzero(targets_t == c, as_tuple=False).flatten()
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        out.extend(perm[:n_per_class].tolist())
    return out


def _parse_hidden_sizes(csv: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    if not parts:
        raise ValueError("icnn_hidden_sizes must be a non-empty comma-separated list (e.g. '512,512,512').")
    try:
        sizes = tuple(int(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"Invalid icnn_hidden_sizes='{csv}'. Expected comma-separated ints.") from e
    if any(s <= 0 for s in sizes):
        raise ValueError(f"Invalid icnn_hidden_sizes='{csv}'. All sizes must be positive.")
    return sizes


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)
    metrics_path = out_dir / args.metrics_file if args.metrics_file else None
    metrics_rows: List[Dict[str, Any]] = []

    def _write_metrics() -> None:
        if metrics_path is None:
            return
        payload = {
            "args": vars(args),
            "metrics": metrics_rows,
        }
        tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(metrics_path)

    vae, vae_ckpt = load_vae(args.vae_ckpt, device)
    theta0, clf_ckpt = load_classifier(args.clf_ckpt, device)

    # Copy θ0 as θ^0 for minimax; θ is updated during GDA.
    theta, _ = load_classifier(args.clf_ckpt, device)

    # Dataset and subset selection: n_per_class samples per class (total n=10*n_per_class).
    ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=_normalize_to_minus1_1())
    indices = make_class_balanced_indices(ds.targets, n_per_class=args.n_per_class, seed=args.seed)
    subset = Subset(ds, indices)

    # Encode subset to latent codes x_i and labels y_i (latent space of the VAE).
    subset_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    x_cpu, y_cpu = encode_subset_to_latents(vae, subset_loader, device)

    x = x_cpu.to(device)
    y = y_cpu.to(device)
    n, d = x.shape
    assert n == 10 * args.n_per_class, f"Expected n=10*n_per_class, got n={n}"
    assert d == 256, f"Expected latent dim d=256 for CIFAR-10, got d={d}"

    # Particle initialization: v_i^0 = x_i, g_i^0 = 0 (Appendix B.2).
    v = x.clone()
    g = torch.zeros_like(v)

    # Transport map T_φ(x,y): residual MLP with label embedding; initialized as identity (Appendix B.2).
    opt_transport = None
    bb_state = None
    bb_armijo_step_params_fn = None
    if args.transport == "mlp":
        tcfg = TransportConfig(latent_dim=d, hidden_width=512, num_classes=10, label_embed_dim=512)
        transport = LabelConditionedTransport(tcfg).to(device)
        opt_transport = Adam(transport.parameters(), lr=args.transport_lr, weight_decay=args.transport_wd)
        transport_cfg_dict: Dict[str, Any] = tcfg.__dict__
    else:
        from wmg.models.icnn import LabelConditionedICNNTransport, ICNNTransportConfig

        hidden_sizes = _parse_hidden_sizes(args.icnn_hidden_sizes)
        icfg = ICNNTransportConfig(
            latent_dim=d,
            hidden_sizes=hidden_sizes,
            num_classes=10,
            label_embed_dim=512,
            activation=args.icnn_activation,
            strong_convexity=args.icnn_strong_convexity,
            nonneg_init=args.icnn_nonneg_init,
            softplus_beta=args.icnn_softplus_beta,
            identity_init=not args.icnn_no_identity_init,
        )
        transport = LabelConditionedICNNTransport(icfg).to(device)
        if args.icnn_bb_armijo:
            from wmg.utils.BB_Armijo import BBArmijoState, bb_armijo_step_params

            bb_armijo_step_params_fn = bb_armijo_step_params
            bb_state = BBArmijoState.create(
                alpha0=args.icnn_alpha0,
                alpha_min=args.icnn_alpha_min,
                alpha_max=args.icnn_alpha_max,
                ls_c=args.icnn_ls_c,
                ls_shrink=args.icnn_ls_shrink,
                ls_max_steps=args.icnn_ls_max_steps,
            )
        else:
            opt_transport = Adam(transport.parameters(), lr=args.transport_lr, weight_decay=args.transport_wd)
        transport_cfg_dict = {**icfg.__dict__, "hidden_sizes": list(icfg.hidden_sizes)}

    # Momentum buffers for θ: h^0 = 0.
    h_params = [torch.zeros_like(p) for p in theta.parameters()]

    # Index loader that cycles through all n samples without repetition each epoch.
    idx_loader = DataLoader(torch.arange(n), batch_size=args.batch_size, shuffle=True, drop_last=True)
    idx_iter = infinite_loader(idx_loader)

    m = args.batch_size
    m_prime = args.match_batch_size
    assert m_prime <= m

    for k in tqdm(range(1, args.k_batches + 1), desc="minimax GDA"):
        log_this = (k % args.log_every) == 0 or k == 1
        metrics_this = (k % args.metrics_every) == 0 or k == 1
        want_metrics = log_this or metrics_this

        idx = next(idx_iter).to(device)

        # ---- (1) Particle + classifier GDA update with momentum (Appendix B.2) ----
        v_batch = v[idx]
        x_batch = x[idx]
        y_batch = y[idx]

        v_var = v_batch.detach().clone().requires_grad_(True)

        logits = theta(v_var)
        ce = F.cross_entropy(logits, y_batch, reduction="none")  # per-sample loss (m,)

        # ℓ_y(θ, v) = CE(θ(v), y) + (ω/2)||θ||^2   (Appendix B.2).
        # We build a sum objective with an m factor on the weight decay so that:
        #   (1/m)∂θ[ce.sum() + m*(ω/2)||θ||^2] = (1/m)Σ∂θ CE_i + ω θ.
        l2 = theta.l2_params()
        loss_sum = ce.sum() + m * 0.5 * args.omega * l2

        theta.zero_grad(set_to_none=True)
        loss_sum.backward()

        grad_v = v_var.grad.detach()  # (m,d), matches ∂_v ℓ(θ, v_i) since loss is sum over i

        # Particle ascent:
        # g_i^{k+1} = ν g_i^k + (∂_v ℓ(θ^k, v_i^k) - (1/γ)(v_i^k - x_i))
        # v_i^{k+1} = v_i^k + η g_i^{k+1}
        g[idx] = args.momentum * g[idx] + (grad_v - (1.0 / args.gamma) * (v_batch - x_batch))
        v[idx] = v_batch + args.eta * g[idx]

        # θ descent:
        # h^{k+1} = ν h^k + (1/m) Σ_i ∂_θ ℓ(θ^k, v_i^k)
        # θ^{k+1} = θ^k - τ h^{k+1}
        with torch.no_grad():
            for (p, h) in zip(theta.parameters(), h_params):
                if p.grad is None:
                    continue
                grad_avg = p.grad.detach() / m
                h.mul_(args.momentum).add_(grad_avg)
                p.add_(h, alpha=-args.tau)

        # ---- (2) Train transport via matching loss (Eq. (22), Alg. 1 step 5) ----
        with torch.no_grad():
            perm = torch.randperm(m, device=device)
            idx_small = idx[perm[:m_prime]]

        x_s = x[idx_small]
        y_s = y[idx_small]
        v_s = v[idx_small].detach()

        match_loss_val: float
        gn_omega: float = float("nan")
        use_adam_transport = (args.transport == "mlp") or ((args.transport == "icnn") and (not args.icnn_bb_armijo))
        if use_adam_transport:
            pred = transport(x_s, y_s)
            match_loss = ((pred - v_s) ** 2).mean()
            match_loss_val = float(match_loss.detach().cpu())

            assert opt_transport is not None
            opt_transport.zero_grad(set_to_none=True)
            match_loss.backward()
            if want_metrics:
                gn = 0.0
                wd = float(args.transport_wd)
                for p in transport.parameters():
                    if p.grad is None:
                        continue
                    g_p = p.grad.detach()
                    if wd != 0.0:
                        g_p = g_p + wd * p.detach()
                    gn += float(g_p.pow(2).sum().cpu())
                gn_omega = gn ** 0.5
            opt_transport.step()
        else:
            assert bb_state is not None
            assert bb_armijo_step_params_fn is not None
            match_loss_val = float("nan")

            def f_params(create_graph: bool) -> torch.Tensor:
                nonlocal match_loss_val
                pred = transport(x_s, y_s, create_graph=create_graph)  # type: ignore[misc]
                mse = ((pred - v_s) ** 2).mean()
                if create_graph and (match_loss_val != match_loss_val):  # NaN check
                    match_loss_val = float(mse.detach().cpu())

                obj = -mse
                if args.transport_wd > 0:
                    l2 = torch.zeros((), device=device)
                    for p in transport.parameters():
                        l2 = l2 + torch.sum(p ** 2)
                    obj = obj - 0.5 * args.transport_wd * l2
                return obj

            _params, bb_state, _f0, gn_omega = bb_armijo_step_params_fn(transport.parameters(), f_params, bb_state)

        # ---- Logging / checkpointing ----
        if log_this:
            with torch.no_grad():
                ce_mean = float(ce.mean().cpu())
                m_loss = float(match_loss_val)
                # Gradient norm of averaged θ-gradient on this batch (optional diagnostic)
                gn_theta = 0.0
                for p in theta.parameters():
                    if p.grad is not None:
                        gn_theta += float((p.grad.detach() / m).pow(2).sum().cpu())
                gn_theta = gn_theta ** 0.5
            tqdm.write(
                f"[k={k:05d}] ce={ce_mean:.4f}  match={m_loss:.6f}  GN_theta={gn_theta:.4f}  GN_omega={gn_omega:.4f}"
            )

        if metrics_this:
            with torch.no_grad():
                ce_mean = float(ce.mean().cpu())
                gn_theta = 0.0
                for p in theta.parameters():
                    if p.grad is not None:
                        gn_theta += float((p.grad.detach() / m).pow(2).sum().cpu())
                gn_theta = gn_theta ** 0.5
            metrics_rows.append({
                "k": int(k),
                "ce": float(ce_mean),
                "match": float(match_loss_val),
                "gn_theta": float(gn_theta),
                "gn_omega": float(gn_omega),
            })

        if (k % args.save_every) == 0 or k == args.k_batches:
            save_checkpoint(out_dir / f"minimax_step{k:06d}.pt", {
                "k": k,
                "args": vars(args),
                "vae_ckpt_path": str(args.vae_ckpt),
                "clf_ckpt_path": str(args.clf_ckpt),
                "classifier_cfg": clf_ckpt.get("clf_cfg", {}),
                "theta_state": theta.state_dict(),
                "theta0_state": theta0.state_dict(),
                "transport_type": str(args.transport),
                "transport_cfg": transport_cfg_dict,
                "transport_state": transport.state_dict(),
                "transport_bb_state": bb_state,
                "x": x.detach().cpu(),
                "y": y.detach().cpu(),
                "v": v.detach().cpu(),
                "g": g.detach().cpu(),
                "h_params": [h.detach().cpu() for h in h_params],
            })

    save_checkpoint(out_dir / "minimax_final.pt", {
        "k": args.k_batches,
        "args": vars(args),
        "vae_ckpt_path": str(args.vae_ckpt),
        "clf_ckpt_path": str(args.clf_ckpt),
        "classifier_cfg": clf_ckpt.get("clf_cfg", {}),
        "theta_state": theta.state_dict(),
        "theta0_state": theta0.state_dict(),
        "transport_type": str(args.transport),
        "transport_cfg": transport_cfg_dict,
        "transport_state": transport.state_dict(),
        "transport_bb_state": bb_state,
        "x": x.detach().cpu(),
        "y": y.detach().cpu(),
        "v": v.detach().cpu(),
        "g": g.detach().cpu(),
        "h_params": [h.detach().cpu() for h in h_params],
    })
    _write_metrics()


if __name__ == "__main__":
    main()
