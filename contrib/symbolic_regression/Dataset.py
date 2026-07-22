import math
import random

try:
    import torch
except Exception:
    torch = None


class Dataset:
    def __init__(
        self,
        X,
        y,
        weights=None,
        *,
        class_labels=None,
        x_units=None,
        y_units=None,
        full_X=None,
        full_y=None,
        full_weights=None,
        full_class_labels=None,
    ):
        self.X = X
        self.y = y
        self.weights = weights
        self.class_labels = class_labels
        self.x_units = x_units
        self.y_units = y_units
        self.full_X = X if full_X is None else full_X
        self.full_y = y if full_y is None else full_y
        self.full_weights = weights if full_weights is None else full_weights
        self.full_class_labels = (
            class_labels if full_class_labels is None else full_class_labels
        )
        self.baseline_loss, self.use_baseline = _compute_baseline_loss(y, weights)

    def with_batch(self, idx):
        Xb = self.full_X[idx]
        yb = self.full_y[idx]
        wb = None if self.full_weights is None else self.full_weights[idx]
        cb = None if self.full_class_labels is None else self.full_class_labels[idx]
        return Dataset(
            Xb,
            yb,
            wb,
            class_labels=cb,
            x_units=self.x_units,
            y_units=self.y_units,
            full_X=self.full_X,
            full_y=self.full_y,
            full_weights=self.full_weights,
            full_class_labels=self.full_class_labels,
        )

    def sample_batch(self, batch_size, rng=None):
        if batch_size <= 0:
            return self
        n = int(self.full_X.shape[0])
        if batch_size >= n:
            return self
        rng = rng or random
        if torch is not None and hasattr(self.full_X, "device"):
            # Use torch indexing to avoid host copies.
            idx = torch.tensor(
                rng.sample(range(n), batch_size),
                dtype=torch.long,
                device=self.full_X.device,
            )
        else:
            idx = rng.sample(range(n), batch_size)
        return self.with_batch(idx)


class State:
    def __init__(
        self,
        last_pops,
        halls_of_fame,
        cur_maxsizes,
        cycles_remaining,
        running_stats=None,
        formatted_hofs=None,
        best_candidates=None,
        options=None,
        run_id=None,
        n_evals=0.0,
        started_at=None,
        ncycles_done=None,
    ):
        self.last_pops = last_pops
        self.halls_of_fame = halls_of_fame
        self.cur_maxsizes = cur_maxsizes
        self.cycles_remaining = cycles_remaining
        self.running_stats = running_stats
        self.formatted_hofs = formatted_hofs
        self.best_candidates = best_candidates
        self.options = options
        self.run_id = run_id
        self.n_evals = n_evals
        self.started_at = started_at
        self.ncycles_done = (
            ncycles_done
            if ncycles_done is not None
            else [[0 for _ in pops] for pops in last_pops]
        )


def _compute_baseline_loss(y, weights=None):
    """
    Baseline loss is MSE of predicting the mean of y (weighted if provided).
    Mirrors SymbolicRegression.jl's baseline normalization.
    """
    if torch is not None and hasattr(y, "detach"):
        with torch.no_grad():
            if weights is None:
                mean_y = y.mean()
                err2 = (y - mean_y) ** 2
                loss_t = err2.mean()
            else:
                w = weights
                mean_y = (y * w).sum() / (w.sum() + 1e-12)
                err2 = (y - mean_y) ** 2
                loss_t = (err2 * w).sum() / (w.sum() + 1e-12)
            loss = float(loss_t.detach().cpu().item())
    else:
        # Fallback: assume y is a sequence
        y_list = list(y)
        if not y_list:
            return 1.0, False
        if weights is None:
            mean_y = sum(y_list) / len(y_list)
            loss = sum((v - mean_y) ** 2 for v in y_list) / len(y_list)
        else:
            w_list = list(weights)
            wsum = sum(w_list) + 1e-12
            mean_y = sum(v * w for v, w in zip(y_list, w_list)) / wsum
            loss = sum(((v - mean_y) ** 2) * w for v, w in zip(y_list, w_list)) / wsum

    if not math.isfinite(loss):
        return 1.0, False
    return loss, True
