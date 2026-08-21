import pytest
import torch

from deepchem.models.symbolic_regression.core.node import BinaryOpNode, ConstantNode, VariableNode
from deepchem.models.symbolic_regression.core.operators import DEFAULT_OPS
from deepchem.models.symbolic_regression.evaluation.evaluate import evaluate_tree
from deepchem.models.symbolic_regression.optimization.optimize import optimize_constants

pytestmark = pytest.mark.torch


def test_optimize_constants_reduces_affine_tree_loss():
    X = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32).unsqueeze(1)
    y = 2.0 * X[:, 0] + 1.0

    # y_hat = c1 * x + c0
    node = BinaryOpNode(
        "add",
        BinaryOpNode("mul", ConstantNode(0.1), VariableNode(0)),
        ConstantNode(-0.5),
    )

    ops = DEFAULT_OPS
    with torch.no_grad():
        before = torch.mean((evaluate_tree(node, X, ops) - y)**2).item()

    optimize_constants(
        node,
        X,
        y,
        ops,
        steps=60,
        lr=0.1,
        n_restarts=1,
        restart_noise=0.2,
    )

    with torch.no_grad():
        after = torch.mean((evaluate_tree(node, X, ops) - y)**2).item()

    assert after < before * 0.1
    assert after < 1e-3


def test_optimize_constants_noop_when_tree_has_no_constants():
    X = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).unsqueeze(1)
    y = X[:, 0]
    node = VariableNode(0)
    out = optimize_constants(node, X, y, DEFAULT_OPS, steps=10)
    assert out is node


def test_optimize_constants_supports_mae_and_huber_objectives():
    X = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32).unsqueeze(1)
    y = 1.5 * X[:, 0] - 0.7
    ops = DEFAULT_OPS

    node_mae = BinaryOpNode(
        "add",
        BinaryOpNode("mul", ConstantNode(-0.2), VariableNode(0)),
        ConstantNode(2.0),
    )
    with torch.no_grad():
        before_mae = torch.mean(torch.abs(evaluate_tree(node_mae, X, ops) -
                                          y)).item()
    optimize_constants(
        node_mae,
        X,
        y,
        ops,
        steps=50,
        lr=0.1,
        loss_type="mae",
    )
    with torch.no_grad():
        after_mae = torch.mean(torch.abs(evaluate_tree(node_mae, X, ops) -
                                         y)).item()
    assert after_mae < before_mae

    node_huber = BinaryOpNode(
        "add",
        BinaryOpNode("mul", ConstantNode(0.3), VariableNode(0)),
        ConstantNode(1.1),
    )
    with torch.no_grad():
        diff_before = torch.abs(evaluate_tree(node_huber, X, ops) - y)
        before_huber = torch.mean(
            torch.where(diff_before <= 1.0, 0.5 * diff_before**2,
                        1.0 * (diff_before - 0.5))).item()
    optimize_constants(
        node_huber,
        X,
        y,
        ops,
        steps=50,
        lr=0.1,
        loss_type="huber",
        huber_delta=1.0,
    )
    with torch.no_grad():
        diff_after = torch.abs(evaluate_tree(node_huber, X, ops) - y)
        after_huber = torch.mean(
            torch.where(diff_after <= 1.0, 0.5 * diff_after**2,
                        1.0 * (diff_after - 0.5))).item()
    assert after_huber < before_huber
