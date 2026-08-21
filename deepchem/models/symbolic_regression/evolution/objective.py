"""Fitness/objective helpers for symbolic regression evolution."""

import math
from typing import Dict

import torch
from torch import Tensor

from deepchem.models.symbolic_regression.core.node import Node
from deepchem.models.symbolic_regression.core.operators import Operator
from deepchem.models.symbolic_regression.evaluation.evaluate import (
    compute_complexity,
    evaluate_tree,
)


def _objective_loss(
    preds: Tensor,
    y_flat: Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> Tensor:
    """Compute scalar objective loss from predictions and targets."""
    if loss_type == "mse":
        return torch.mean((preds - y_flat)**2)
    if loss_type == "mae":
        return torch.mean(torch.abs(preds - y_flat))
    if loss_type == "huber":
        diff = torch.abs(preds - y_flat)
        quadratic = 0.5 * diff**2
        linear = huber_delta * (diff - 0.5 * huber_delta)
        return torch.mean(torch.where(diff <= huber_delta, quadratic, linear))
    raise ValueError(
        f"loss_type must be one of ('mse', 'mae', 'huber'), got {loss_type!r}")


def fit_linear_scaling(
    preds: Tensor,
    y_flat: Tensor,
    eps: float = 1e-12,
) -> tuple:
    """Fit affine scaling y_hat' = a * y_hat + b to reduce MSE.

    Returns scaled predictions and fitted (a, b). If fitting is degenerate or
    unstable, returns identity scaling.
    """
    pred_flat = preds.flatten()
    target_flat = y_flat.flatten()
    if pred_flat.numel() == 0 or target_flat.numel() == 0:
        return pred_flat, 1.0, 0.0

    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)
    centered_preds = pred_flat - pred_mean
    denom = torch.mean(centered_preds * centered_preds)

    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        a = torch.tensor(0.0, dtype=pred_flat.dtype, device=pred_flat.device)
        b = target_mean
    else:
        covariance = torch.mean(centered_preds * (target_flat - target_mean))
        a = covariance / denom
        b = target_mean - a * pred_mean

    if not torch.isfinite(a) or not torch.isfinite(b):
        return pred_flat, 1.0, 0.0

    scaled = a * pred_flat + b
    scaled = torch.nan_to_num(scaled, nan=1e6, posinf=1e6, neginf=-1e6)
    return scaled, float(a.item()), float(b.item())


def score_tree(node: Node,
               X: Tensor,
               y: Tensor,
               operators: Dict[str, Operator],
               parsimony_coefficient: float,
               loss_type: str = "mse",
               loss_scale: str = "linear",
               huber_delta: float = 1.0,
               loss_epsilon: float = 1e-12) -> tuple:
    """Evaluate tree and return (mse, objective_loss, fitness, complexity)."""
    with torch.no_grad():
        preds = evaluate_tree(node, X, operators)
        preds = torch.nan_to_num(preds, nan=1e6, posinf=1e6, neginf=-1e6)
        y_flat = y.flatten()
        mse = torch.mean((preds - y_flat)**2).item()
        objective_loss = _objective_loss(preds,
                                         y_flat,
                                         loss_type=loss_type,
                                         huber_delta=huber_delta).item()

    cplx = compute_complexity(node, operators)
    fitness = fitness_from_loss_complexity(objective_loss,
                                           cplx,
                                           parsimony_coefficient,
                                           loss_scale=loss_scale,
                                           loss_epsilon=loss_epsilon)
    return mse, objective_loss, fitness, cplx


def fitness_from_loss_complexity(
    loss: float,
    complexity: int,
    parsimony_coefficient: float,
    loss_scale: str = "linear",
    loss_epsilon: float = 1e-12,
) -> float:
    """Compute fitness from cached objective-loss and complexity values."""
    if loss_scale == "linear":
        loss_term = loss
    elif loss_scale == "log":
        loss_term = math.log(max(loss, loss_epsilon))
    else:
        raise ValueError(
            f"loss_scale must be 'linear' or 'log', got {loss_scale!r}")
    return loss_term + parsimony_coefficient * complexity
