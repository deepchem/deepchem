import random

import pytest

from deepchem.models.symbolic_regression.core.node import (
    BinaryOpNode,
    ConstantNode,
    UnaryOpNode,
    VariableNode,
)
from deepchem.models.symbolic_regression.core.operators import DEFAULT_BINARY_OPS, DEFAULT_UNARY_OPS
from deepchem.models.symbolic_regression.genetic.mutation import (
    get_default_mutation_weights,
    mutate_with_info,
    validate_custom_mutation_weights,
)
from deepchem.models.symbolic_regression.genetic.selection import tournament_select

pytestmark = pytest.mark.torch


def test_validate_custom_mutation_weights_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_custom_mutation_weights({"add_node": -1.0})

    with pytest.raises(ValueError):
        validate_custom_mutation_weights({"unknown_name": 1.0})


def test_mutate_with_delete_weight_returns_smaller():
    node = BinaryOpNode("add", ConstantNode(0.0), VariableNode(0))
    out, name = mutate_with_info(
        node.copy(),
        DEFAULT_UNARY_OPS,
        DEFAULT_BINARY_OPS,
        n_features=1,
        weights={"delete_node": 1.0},
        rng=random.Random(0),
    )
    assert name == "delete_node"


def test_mutate_with_info_reports_applied_mutation():
    node = BinaryOpNode("add", VariableNode(0), ConstantNode(1.0))
    out, name = mutate_with_info(
        node.copy(),
        DEFAULT_UNARY_OPS,
        DEFAULT_BINARY_OPS,
        n_features=1,
        weights={"mutate_operator": 1.0},
        rng=random.Random(0),
    )
    assert name == "mutate_operator"
    assert "mutate_operator" in get_default_mutation_weights()


def test_swap_operands_only_swaps_non_commutative():
    """swap_operands now always skips commutative ops (add, mul)."""
    node = BinaryOpNode("sub", VariableNode(0), VariableNode(1))

    out, _ = mutate_with_info(
        node.copy(),
        DEFAULT_UNARY_OPS,
        DEFAULT_BINARY_OPS,
        n_features=2,
        weights={"swap_operands": 1.0},
        rng=random.Random(0),
    )
    # sub is non-commutative, so swap should happen
    assert str(out) == "sub(x1, x0)"

    # add is commutative — swap should NOT happen
    add_node = BinaryOpNode("add", VariableNode(0), VariableNode(1))
    out2, _ = mutate_with_info(
        add_node.copy(),
        DEFAULT_UNARY_OPS,
        DEFAULT_BINARY_OPS,
        n_features=2,
        weights={"swap_operands": 1.0},
        rng=random.Random(0),
    )
    assert str(out2) == "add(x0, x1)"


def test_tournament_select_picks_best_with_full_tournament():
    population = [VariableNode(0), VariableNode(1), VariableNode(2)]
    fitnesses = [1.0, 0.1, 0.5]
    selected = tournament_select(
        population,
        fitnesses,
        tournament_size=3,
        pick_prob=1.0,
        rng=random.Random(123),
    )
    assert isinstance(selected, VariableNode)
    assert selected.index == 1


def test_optimize_constants_mutation_uses_optimizer_context():
    node = BinaryOpNode("add", ConstantNode(1.0), VariableNode(0))
    seen = {}

    def fake_optimize(tree, **kwargs):
        seen["tree"] = tree
        seen["kwargs"] = kwargs
        return BinaryOpNode("add", ConstantNode(9.0), VariableNode(0))

    out, name = mutate_with_info(
        node.copy(),
        DEFAULT_UNARY_OPS,
        DEFAULT_BINARY_OPS,
        n_features=1,
        weights={"optimize_constants": 1.0},
        optimize_fn=fake_optimize,
        optimize_kwargs={
            "X": "X",
            "y": "y",
            "operators": {}
        },
        rng=random.Random(0),
    )
    assert name == "optimize_constants"
    assert isinstance(out, BinaryOpNode)
    assert isinstance(out.left, ConstantNode)
    assert float(out.left.value.item()) == pytest.approx(9.0)
    assert seen["kwargs"]["X"] == "X"
    assert seen["kwargs"]["y"] == "y"


def test_append_argument_mutation_wraps_operator_argument():
    node = UnaryOpNode("sin", VariableNode(0))
    out, name = mutate_with_info(
        node.copy(),
        {"sin": DEFAULT_UNARY_OPS["sin"]},
        {"add": DEFAULT_BINARY_OPS["add"]},
        n_features=1,
        weights={"append_argument": 1.0},
        rng=random.Random(0),
    )
    assert name == "append_argument"
    assert isinstance(out, UnaryOpNode)
    assert isinstance(out.child, BinaryOpNode)
    assert out.child.op_name == "add"
