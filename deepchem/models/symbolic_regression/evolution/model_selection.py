"""Final-equation selection policies for symbolic regression."""

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor

from deepchem.models.symbolic_regression.core.operators import Operator
from deepchem.models.symbolic_regression.evaluation.evaluate import (
    compute_complexity,
    evaluate_tree,
)
from deepchem.models.symbolic_regression.evolution.hall_of_fame import HoFEntry

SelectionMethod = Literal["accuracy", "score", "best"]


@dataclass
class CandidateStats:
    """Validation stats for one Hall-of-Fame candidate."""

    entry: HoFEntry
    complexity: int
    val_mse: float
    score: float = float("-inf")


def evaluate_hof_on_validation(
    entries: List[HoFEntry],
    X_val: Tensor,
    y_val: Tensor,
    operators: Dict[str, Operator],
) -> List[CandidateStats]:
    """Evaluate Hall-of-Fame candidates on validation data."""
    if not entries:
        return []

    y_flat = y_val.flatten()
    stats: List[CandidateStats] = []

    with torch.no_grad():
        for entry in entries:
            preds = evaluate_tree(entry.node, X_val, operators)
            preds = torch.nan_to_num(preds, nan=1e6, posinf=1e6, neginf=-1e6)
            mse = torch.mean((preds - y_flat)**2).item()
            complexity = (entry.complexity if entry.complexity >= 0 else
                          compute_complexity(entry.node, operators))
            stats.append(
                CandidateStats(entry=entry, complexity=complexity, val_mse=mse))

    stats.sort(key=lambda s: (s.complexity, s.val_mse))
    return stats


def add_complexity_scores(stats: List[CandidateStats],
                          eps: float = 1e-12) -> None:
    """Populate complexity-improvement scores in-place.

    Score mirrors PySR-style preference for equations that gain substantial
    accuracy with small complexity increases.
    """
    if not stats:
        return

    stats[0].score = float("-inf")
    for i in range(1, len(stats)):
        prev = stats[i - 1]
        cur = stats[i]
        dc = max(1, cur.complexity - prev.complexity)
        prev_loss = max(prev.val_mse, eps)
        cur_loss = max(cur.val_mse, eps)
        cur.score = (math.log(prev_loss) - math.log(cur_loss)) / dc


def select_entry(
    stats: List[CandidateStats],
    method: SelectionMethod = "best",
    best_loss_factor: float = 1.5,
) -> Optional[HoFEntry]:
    """Select final equation from validation stats."""
    if not stats:
        return None

    add_complexity_scores(stats)

    if method == "accuracy":
        return min(stats, key=lambda s: s.val_mse).entry

    if method == "score":
        best_scored = max(stats, key=lambda s: s.score)
        # If complexity score does not indicate a meaningful tradeoff,
        # default to pure validation accuracy.
        if (best_scored.score == float("-inf") or
                not math.isfinite(best_scored.score) or
                best_scored.score <= 0.0):
            return min(stats, key=lambda s: s.val_mse).entry
        return best_scored.entry

    if method == "best":
        min_loss = min(s.val_mse for s in stats)
        threshold = best_loss_factor * min_loss
        filtered = [s for s in stats if s.val_mse <= threshold]
        if not filtered:
            return min(stats, key=lambda s: s.val_mse).entry
        best_scored = max(filtered, key=lambda s: s.score)
        # Same guard as "score": when score is not informative,
        # prefer the lowest validation loss among eligible candidates.
        if (best_scored.score == float("-inf") or
                not math.isfinite(best_scored.score) or
                best_scored.score <= 0.0):
            return min(filtered, key=lambda s: s.val_mse).entry
        return best_scored.entry

    raise ValueError(f"Unknown selection method: {method}. Expected one of: "
                     "'accuracy', 'score', 'best'.")
