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
#  BB + Armijo line search (for ascent)
# ============================================================

@dataclass
class BBArmijoState:
    alpha_min: float
    alpha_max: float
    alpha_prev: float
    ls_c: float
    ls_shrink: float
    ls_max_steps: int
    prev_params_vec: Optional[torch.Tensor] = None
    prev_grad_vec: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        alpha0: float = 1e-1,
        alpha_min: float = 1e-6,
        alpha_max: float = 10.0,
        ls_c: float = 1e-4,
        ls_shrink: float = 0.5,
        ls_max_steps: int = 10,
    ) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(
            alpha_min=float(max(alpha_min, 1e-12)),
            alpha_max=float(max(alpha_max, alpha_min)),
            alpha_prev=alpha0,
            ls_c=float(ls_c),
            ls_shrink=float(ls_shrink),
            ls_max_steps=int(max(ls_max_steps, 1)),
        )

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
        """Propose a BB step size (no Armijo yet)."""
        if (
            self.prev_params_vec is None
            or self.prev_grad_vec is None
            or self.prev_params_vec.shape != params_vec.shape
            or self.prev_grad_vec.shape != grad_vec.shape
        ):
            alpha = self.alpha_prev
        else:
            s = params_vec - self.prev_params_vec
            y = grad_vec - self.prev_grad_vec
            denom = torch.dot(s, y)
            num = torch.dot(s, s)
            cond = torch.isfinite(denom) & (torch.abs(denom) > 1e-12)
            alpha_bb = torch.where(cond, num / denom, torch.tensor(self.alpha_prev, device=denom.device))
            alpha = float(alpha_bb.clamp(self.alpha_min, self.alpha_max).item())

        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        alpha = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return alpha

    def update_history(self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float) -> "BBArmijoState":
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return BBArmijoState(
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_prev=alpha_clamped,
            ls_c=self.ls_c,
            ls_shrink=self.ls_shrink,
            ls_max_steps=self.ls_max_steps,
            prev_params_vec=params_vec.detach().clone(),
            prev_grad_vec=grad_vec.detach().clone(),
        )


def bb_armijo_ascent_x(
    x0: torch.Tensor,
    f,
    num_steps: int,
    bb_state: Optional[BBArmijoState] = None,
) -> Tuple[torch.Tensor, BBArmijoState, Optional[torch.Tensor]]:
    """
    Gradient ascent on x with BB step size + Armijo backtracking.

    f: x -> scalar objective (maximized).
    """
    if x0.numel() == 0 or num_steps == 0:
        state = bb_state if bb_state is not None else BBArmijoState.create()
        return x0, state, None

    state = bb_state if bb_state is not None else BBArmijoState.create()

    x = x0
    last_grad = None
    for _ in range(num_steps):
        x_req = x.detach().requires_grad_(True)
        fx = f(x_req)
        g = torch.autograd.grad(fx, x_req, create_graph=False, retain_graph=False)[0]
        g_vec = g.reshape(-1).detach()
        x_vec = x_req.detach().reshape(-1)
        alpha = state.propose(x_vec, g_vec)

        fx_val = float(fx.detach())
        g_dot_g = float(torch.dot(g_vec, g_vec).item())
        if g_dot_g == 0.0:
            break

        alpha_k = alpha
        for _ in range(state.ls_max_steps):
            x_trial = x_req + alpha_k * g
            f_trial = float(f(x_trial).detach())
            if f_trial >= fx_val + state.ls_c * alpha_k * g_dot_g:
                break
            alpha_k *= state.ls_shrink

        x_new = (x_req + alpha_k * g).detach()
        x_new_req = x_new.requires_grad_(True)
        g_new = torch.autograd.grad(f(x_new_req), x_new_req, create_graph=False, retain_graph=False)[0]
        state = state.update_history(x_new_req.detach().reshape(-1), g_new.detach().reshape(-1), alpha_k)
        x = x_new.detach()
        last_grad = g_new.detach()

    return x, state, last_grad


def bb_armijo_step_params(
    params: Any,
    f_params,
    bb_state: BBArmijoState,
) -> Tuple[Any, BBArmijoState, float, float]:
    """
    Single BB+Armijo gradient-ascent step on a parameter collection.

    f_params: callable taking a boolean `create_graph` flag and returning scalar objective (maximized).
    """
    params = list(params)
    params_vec = nn_utils.parameters_to_vector(params).detach()

    f_val = f_params(True)
    grads = torch.autograd.grad(
        f_val,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grad_tensors = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads)]
    grad_vec = nn_utils.parameters_to_vector(grad_tensors)
    grad_norm = grad_vec.norm().item()

    alpha = bb_state.propose(params_vec, grad_vec)
    f_val_float = float(f_val.detach())
    g_dot_g = float(torch.dot(grad_vec, grad_vec).item())
    if g_dot_g == 0.0:
        return params, bb_state, f_val_float, grad_norm

    alpha_k = alpha
    for _ in range(bb_state.ls_max_steps):
        trial_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(trial_vec, params)
            f_trial = float(f_params(False).detach())
        if f_trial >= f_val_float + bb_state.ls_c * alpha_k * g_dot_g:
            break
        alpha_k *= bb_state.ls_shrink

    final_vec = params_vec + alpha_k * grad_vec
    with torch.no_grad():
        nn_utils.vector_to_parameters(final_vec, params)

    f_new = f_params(True)
    grads_new = torch.autograd.grad(
        f_new,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grad_tensors_new = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads_new)]
    grad_vec_new = nn_utils.parameters_to_vector(grad_tensors_new)
    new_state = bb_state.update_history(final_vec, grad_vec_new, alpha_k)
    return params, new_state, f_val_float, grad_vec_new.norm().item()