import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import matplotlib.pyplot as plt
from matplotlib import ticker
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from contextlib import nullcontext



# ============================================================
#  ICNN components (Input-Convex Potential φ_ψ)
# ============================================================


def icnn_principled_moments(fan_in: int):
    """Principled log-normal moments for positive weights."""
    if fan_in <= 0:
        raise ValueError(f"ICNN fan-in must be positive; got {fan_in}.")
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma2 = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    tilde_sigma = math.sqrt(tilde_sigma2)
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


class NonNegativeLinear(nn.Module):
    """Linear map with strictly non-negative weights via exp/softplus."""
    def __init__(self, in_features, out_features, bias=True, init_mode="principled"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.init_mode = init_mode.lower()
        if self.init_mode not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported init_mode '{init_mode}' for NonNegativeLinear.")
        self.parametrization = "exp" if self.init_mode == "principled" else "softplus"

        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(self.in_features)
            with torch.no_grad():
                if tilde_sigma == 0.0:
                    self.weight_param.fill_(tilde_mu)
                else:
                    self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
                if self.bias is not None:
                    self.bias.fill_(mu_b)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.parametrization == "exp":
            weight = torch.exp(self.weight_param)
        else:
            weight = F.softplus(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class InputConvexPotential(nn.Module):
    """
    Dense ICNN potential φ(z):
      φ(z) = 0.5 μ ||z||² + aᵀ z + g(z; ψ)
    with non-negative couplings for convexity.
    """
    def __init__(
        self,
        input_dim=2,
        hidden_sizes=(64, 64, 64),
        activation="softplus",
        strong_convexity=1.0,
        nonneg_init="principled",
        softplus_beta=20.0,
    ):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.strong_convexity = strong_convexity
        self.softplus_beta = softplus_beta

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        for i, width in enumerate(hidden_sizes):
            self.z_linears.append(nn.Linear(input_dim, width, bias=True))
            if i == 0:
                self.h_linears.append(None)
            else:
                self.h_linears.append(
                    NonNegativeLinear(
                        hidden_sizes[i - 1],
                        width,
                        bias=False,
                        init_mode=nonneg_init,
                    )
                )

        self.hidden_output = NonNegativeLinear(
            hidden_sizes[-1],
            1,
            bias=True,
            init_mode=nonneg_init,
        )
        self.input_skip = nn.Linear(input_dim, 1, bias=True)

    def _activation(self):
        act_name = self.activation.lower()
        if act_name == "relu":
            return F.relu
        if act_name == "softplus":
            beta = float(self.softplus_beta)
            return lambda u: F.softplus(beta * u) / beta
        raise ValueError(f"Unsupported ICNN activation '{self.activation}'.")

    def forward(self, x):
        """
        x: (n, d); returns scalar potential φ(x) per sample.
        """
        z = x.view(x.size(0), -1)
        act = self._activation()

        h = act(self.z_linears[0](z))
        for k in range(1, len(self.z_linears)):
            z_term = self.z_linears[k](z)
            h_term = self.h_linears[k](h) if self.h_linears[k] is not None else 0.0
            h = act(z_term + h_term)

        quadratic = 0.5 * self.strong_convexity * (z ** 2).sum(dim=1, keepdim=True)
        out = quadratic + self.input_skip(z) + self.hidden_output(h)
        return out.squeeze(-1)


ICNN = InputConvexPotential


def icnn_transport(icnn, x, create_graph=False):
    """
    Compute T_ψ(x) = ∇_x φ_ψ(x) via autograd.
    Returns T_ψ(x) with same shape as x.
    """
    x_in = x.clone().detach().requires_grad_(True)
    # Ensure grad tracking even if called under torch.no_grad()
    with torch.set_grad_enabled(True):
        phi = icnn(x_in)
        grads = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=x_in,
            create_graph=create_graph,
        )[0]
    return grads.view_as(x)






@dataclass(frozen=True)
class ICNNTransportConfig:
    latent_dim: int = 256
    hidden_sizes: Tuple[int, ...] = (512, 512, 512)
    num_classes: int = 10
    label_embed_dim: int = 512
    activation: str = "softplus"
    strong_convexity: float = 1.0
    nonneg_init: str = "principled"
    softplus_beta: float = 20.0
    identity_init: bool = True


class LabelConditionedICNNTransport(nn.Module):
    """Label-conditioned ICNN transport map in latent space.

    Implements:
        T(x, y) = ∇_x φ([x; emb(y)])
    where φ is an ICNN potential convex in its input.
    """

    def __init__(self, cfg: ICNNTransportConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.num_classes, cfg.label_embed_dim)
        self.potential = InputConvexPotential(
            input_dim=cfg.latent_dim + cfg.label_embed_dim,
            hidden_sizes=cfg.hidden_sizes,
            activation=cfg.activation,
            strong_convexity=cfg.strong_convexity,
            nonneg_init=cfg.nonneg_init,
            softplus_beta=cfg.softplus_beta,
        )
        if cfg.identity_init:
            self._init_identity()

    def _init_identity(self) -> None:
        """Initialize so T(x,y) ≈ x (when strong_convexity=1.0)."""
        with torch.no_grad():
            nn.init.zeros_(self.embed.weight)
            nn.init.zeros_(self.potential.input_skip.weight)
            if self.potential.input_skip.bias is not None:
                nn.init.zeros_(self.potential.input_skip.bias)
            for lin in self.potential.z_linears:
                nn.init.zeros_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            for hlin in self.potential.h_linears:
                if hlin is None:
                    continue
                hlin.weight_param.zero_()
            self.potential.hidden_output.weight_param.zero_()
            if self.potential.hidden_output.bias is not None:
                self.potential.hidden_output.bias.zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        x_in = x.detach().requires_grad_(True)
        with torch.set_grad_enabled(True):
            ey = self.embed(y)
            z = torch.cat([x_in, ey], dim=1)
            phi = self.potential(z)
            grads = torch.autograd.grad(
                outputs=phi.sum(),
                inputs=x_in,
                create_graph=create_graph,
            )[0]
        return grads.view_as(x)


def parameter_grad_norm(parameters):
    sq_norms = []
    for p in parameters:
        if p.grad is not None:
            sq_norms.append(p.grad.detach().norm() ** 2)
    if not sq_norms:
        return 0.0
    return torch.sqrt(torch.stack(sq_norms).sum()).item()
