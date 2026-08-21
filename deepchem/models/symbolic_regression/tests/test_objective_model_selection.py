import pytest
import torch

from deepchem.models.symbolic_regression.core.node import BinaryOpNode, UnaryOpNode, VariableNode
from deepchem.models.symbolic_regression.core.operators import DEFAULT_OPS
from deepchem.models.symbolic_regression.evolution.hall_of_fame import HoFEntry
from deepchem.models.symbolic_regression.evolution.model_selection import (
    add_complexity_scores,
    evaluate_hof_on_validation,
    select_entry,
)
from deepchem.models.symbolic_regression.evolution.objective import score_tree

pytestmark = pytest.mark.torch


def test_score_tree_returns_finite_values_with_protected_ops():
    ops = DEFAULT_OPS
    node = UnaryOpNode("log", VariableNode(0))
    X = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
    y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    mse, loss, fitness, complexity = score_tree(
        node,
        X,
        y,
        ops,
        parsimony_coefficient=0.001,
        loss_type="mse",
        loss_scale="linear",
    )
    assert torch.isfinite(torch.tensor(mse))
    assert torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(torch.tensor(fitness))
    assert complexity > 0


def test_model_selection_prefers_accurate_entry_when_scores_not_informative():
    ops = DEFAULT_OPS
    X = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).unsqueeze(1)
    y = X[:, 0]

    entries = [
        HoFEntry(node=VariableNode(0), mse=0.0, fitness=0.0, complexity=1),
        HoFEntry(
            node=BinaryOpNode("add", VariableNode(0), VariableNode(0)),
            mse=1.0,
            fitness=1.0,
            complexity=3,
        ),
    ]

    stats = evaluate_hof_on_validation(entries, X, y, ops)
    add_complexity_scores(stats)
    chosen = select_entry(stats, method="score")
    assert chosen is not None
    assert str(chosen.node) == "x0"
