import torch
from DimensionalAnalysis import has_dimensional_mismatch, dimensional_mismatch_score
import math

def _weighted_mean(loss_vec, w):
    if w is None:
        return loss_vec.mean()
    return (loss_vec * w).sum() / (w.sum() + 1e-12)

def _builtin_loss(dataset, tree, options):
    X = dataset.X
    y = dataset.y
    w = getattr(dataset, "weights", None)
    yhat = tree.forward(X)
    if torch.isnan(yhat).any() or torch.isinf(yhat).any():
        return float("inf")

    name = str(
        getattr(options, "loss_name", "mse") if options is not None else "mse"
    ).lower()
    err = yhat - y

    if name == "mse":
        loss_t = err.square()
    elif name == "mae":
        loss_t = torch.abs(err)
    elif name == "huber":
        delta = float(getattr(options, "huber_delta", 1.0))
        abs_err = torch.abs(err)
        quad = torch.minimum(
            abs_err, torch.tensor(delta, device=abs_err.device, dtype=abs_err.dtype)
        )
        lin = abs_err - quad
        loss_t = 0.5 * quad.square() + delta * lin
    elif name == "logcosh":
        loss_t = torch.log(torch.cosh(torch.clamp(err, -20.0, 20.0)))
    elif name == "quantile":
        q = float(getattr(options, "quantile", 0.5))
        q = min(1.0, max(0.0, q))
        u = y - yhat
        loss_t = torch.maximum(q * u, (q - 1.0) * u)
    elif name == "hinge":
        y_signed = torch.where(y > 0.0, torch.ones_like(y), -torch.ones_like(y))
        loss_t = torch.relu(1.0 - y_signed * yhat)
    elif name == "bce":
        p = torch.sigmoid(yhat)
        p = torch.clamp(p, 1e-8, 1.0 - 1e-8)
        loss_t = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    else:
        # Fallback to MSE on unknown loss names.
        loss_t = err.square()

    loss_m = _weighted_mean(loss_t, w)
    return float(loss_m.detach().cpu().item())


def _normalization(dataset):
    baseline = getattr(dataset, "baseline_loss", None)
    use_baseline = bool(getattr(dataset, "use_baseline", False))
    if baseline is None or not math.isfinite(float(baseline)):
        use_baseline = False
    if use_baseline and float(baseline) >= 0.01:
        return float(baseline)
    return 0.01


def calculate_loss_and_cost(
    complexity,
    dataset,
    tree,
    parsimony_penalty,
    options=None,
):
    custom_loss = (
        getattr(options, "loss_function", None) if options is not None else None
    )
    expression_loss = (
        getattr(options, "loss_function_expression", None)
        if options is not None
        else None
    )
    if custom_loss is not None:
        try:
            loss = float(custom_loss(tree, dataset, options))
        except TypeError:
            loss = float(custom_loss(tree, dataset))
    elif expression_loss is not None:
        yhat = tree.forward(dataset.X)
        if torch.isnan(yhat).any() or torch.isinf(yhat).any():
            return float("inf"), float("inf")
        try:
            loss = float(expression_loss(yhat, dataset, tree, options))
        except TypeError:
            try:
                loss = float(expression_loss(yhat, dataset))
            except TypeError:
                loss = float(expression_loss(yhat, dataset.y))
    else:
        loss = _builtin_loss(dataset, tree, options)

    if math.isnan(loss) or math.isinf(loss):
        return float("inf"), float("inf")

    loss_scale = (
        str(getattr(options, "loss_scale", "log")) if options is not None else "log"
    )
    if loss_scale == "log" and loss < 0.0:
        return float("inf"), float("inf")

    norm = _normalization(dataset)
    cost = float((loss / norm) + parsimony_penalty * complexity)
    dim_penalty = (
        getattr(options, "dimensional_constraint_penalty", None)
        if options is not None
        else None
    )
    enforce_dims = bool(
        getattr(options, "enforce_dimensional_constraints", False)
        if options is not None
        else False
    )
    if dim_penalty is not None or enforce_dims:
        mismatch = has_dimensional_mismatch(tree, dataset, options)
        if mismatch and enforce_dims:
            return float("inf"), float("inf")
        if mismatch and dim_penalty is not None:
            cost += float(dim_penalty) * dimensional_mismatch_score(
                tree, dataset, options
            )

    if math.isnan(cost) or math.isinf(cost):
        return float("inf"), float("inf")
    return loss, cost