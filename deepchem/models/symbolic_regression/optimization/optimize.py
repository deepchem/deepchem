"""Constant optimization for symbolic regression.

Uses L-BFGS with strong Wolfe line search — converges fast for
the small problems typical in SR (1–5 constants per tree).

Optimization is restart-aware and keeps the best trial only, so constants
never regress because of a bad local optimization step.
"""

from typing import Dict, Literal

import torch
from torch import Tensor

from deepchem.models.symbolic_regression.core.node import Node, get_constants
from deepchem.models.symbolic_regression.core.operators import Operator
from deepchem.models.symbolic_regression.evaluation.evaluate import evaluate_tree
from deepchem.models.symbolic_regression.evolution.objective import _objective_loss


def _compute_loss(node: Node,
                  X: Tensor,
                  y: Tensor,
                  operators: Dict[str, Operator],
                  loss_type: str = "mse",
                  huber_delta: float = 1.0) -> Tensor:
    """Compute objective loss for the current tree."""
    predictions = evaluate_tree(node, X, operators)
    predictions = torch.nan_to_num(predictions,
                                   nan=1e6,
                                   posinf=1e6,
                                   neginf=-1e6)
    return _objective_loss(predictions,
                           y.flatten(),
                           loss_type=loss_type,
                           huber_delta=huber_delta)


def optimize_constants(node: Node,
                       X: Tensor,
                       y: Tensor,
                       operators: Dict[str, Operator],
                       steps: int = 10,
                       lr: float = 0.01,
                       loss_type: Literal["mse", "mae", "huber"] = "mse",
                       huber_delta: float = 1.0,
                       n_restarts: int = 0,
                       restart_noise: float = 0.1) -> Node:
    """Optimize constants in an expression tree via L-BFGS.

    Extracts all ConstantNodes, replaces their values with
    torch.nn.Parameters, optimizes them to minimize configured loss, then
    writes the optimized values back as plain tensors.

    Parameters
    ----------
    node : Node
        Root of the expression tree (modified in-place).
    X : Tensor
        Input data of shape (n_samples, n_features).
    y : Tensor
        Target values of shape (n_samples,).
    operators : dict[str, Operator]
        Operator registry mapping names to Operator objects.
    steps : int
        Number of optimization steps. Default 10.
    lr : float
        Learning rate. Default 0.01.
    loss_type : {"mse", "mae", "huber"}
        Objective minimized by constant optimization.
    huber_delta : float
        Delta used only when `loss_type="huber"`.
    n_restarts : int
        Number of random restarts after the initial run. Total trials are
        `1 + n_restarts`. Default 0.
    restart_noise : float
        Standard deviation of Gaussian perturbation added to the starting
        constants for restart trials. Default 0.1.
    Returns
    -------
    Node
        The same tree with optimized constant values.
    """
    if n_restarts < 0:
        raise ValueError(f"n_restarts must be >= 0, got {n_restarts}")
    if restart_noise < 0:
        raise ValueError(f"restart_noise must be >= 0, got {restart_noise}")
    if loss_type not in ("mse", "mae", "huber"):
        raise ValueError(
            "loss_type must be one of ('mse', 'mae', 'huber'), got "
            f"{loss_type!r}")
    if huber_delta <= 0:
        raise ValueError(f"huber_delta must be > 0, got {huber_delta}")

    constants = get_constants(node)

    # Nothing to optimize
    if not constants:
        return node

    # Save original values and make sure they match compute dtype/device.
    original_values = [
        c.value.clone().detach().to(dtype=X.dtype, device=X.device)
        for c in constants
    ]
    best_values = [v.clone() for v in original_values]
    with torch.no_grad():
        loss = _compute_loss(node,
                             X,
                             y,
                             operators,
                             loss_type=loss_type,
                             huber_delta=huber_delta)
    best_loss = float("inf") if not torch.isfinite(loss) else float(loss.item())

    for restart_idx in range(n_restarts + 1):
        trial_values = []
        for orig in original_values:
            if restart_idx == 0 or restart_noise == 0.0:
                trial = orig.clone()
            else:
                trial = orig + restart_noise * torch.randn_like(orig)
            trial_values.append(trial)

        params = []
        for c, trial in zip(constants, trial_values):
            param = torch.nn.Parameter(trial.clone().detach())
            c.value = param
            params.append(param)

        try:
            _optimize_bfgs(
                node,
                X,
                y,
                operators,
                params,
                steps,
                lr,
                loss_type=loss_type,
                huber_delta=huber_delta,
            )
            with torch.no_grad():
                trial_loss_t = _compute_loss(
                    node,
                    X,
                    y,
                    operators,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
            trial_loss = (float("inf") if not torch.isfinite(trial_loss_t) else
                          float(trial_loss_t.item()))

            if trial_loss < best_loss:
                best_loss = trial_loss
                best_values = [p.detach().clone() for p in params]
        except (RuntimeError, ValueError):
            # Keep searching; we'll restore the best known constants below.
            continue

    for c, best in zip(constants, best_values):
        c.value = best.detach().clone()

    return node


def _optimize_bfgs(node,
                   X,
                   y,
                   operators,
                   params,
                   steps,
                   lr,
                   loss_type: str = "mse",
                   huber_delta: float = 1.0):
    """L-BFGS optimization with closure and strong Wolfe line search."""
    opt = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=steps,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = _compute_loss(
            node,
            X,
            y,
            operators,
            loss_type=loss_type,
            huber_delta=huber_delta,
        )

        if not torch.isfinite(loss):
            return loss

        loss.backward()

        # Clip gradients to prevent explosions from safe_exp / safe_log
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)

        return loss

    # LBFGS does all iterations internally via the closure
    opt.step(closure)
