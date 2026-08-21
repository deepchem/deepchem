import pytest
import torch

from deepchem.models.symbolic_regression.core.node import BinaryOpNode, ConstantNode, UnaryOpNode, VariableNode
from deepchem.models.symbolic_regression.core.operators import DEFAULT_OPS, safe_div, safe_exp, safe_log
from deepchem.models.symbolic_regression.evaluation.evaluate import compute_complexity, evaluate_tree

pytestmark = pytest.mark.torch


def test_safe_div_handles_small_denominator():
    a = torch.tensor([2.0, 4.0], dtype=torch.float32)
    b = torch.tensor([1e-12, 2.0], dtype=torch.float32)
    out = safe_div(a, b)
    assert torch.allclose(out, torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_safe_log_and_safe_exp_are_finite():
    x_log = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
    out_log = safe_log(x_log)
    assert torch.isfinite(out_log).all()

    x_exp = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float32)
    out_exp = safe_exp(x_exp)
    assert torch.isfinite(out_exp).all()
    assert torch.allclose(
        out_exp,
        torch.tensor([
            torch.exp(torch.tensor(-10.0)), 1.0,
            torch.exp(torch.tensor(10.0))
        ]),
        atol=1e-6,
    )


def test_evaluate_tree_matches_manual_expression():
    # (x0 + 2) * x1
    node = BinaryOpNode(
        "mul",
        BinaryOpNode("add", VariableNode(0), ConstantNode(2.0)),
        VariableNode(1),
    )
    X = torch.tensor([[1.0, 3.0], [2.0, 4.0], [-1.0, 5.0]], dtype=torch.float32)
    out = evaluate_tree(node, X, DEFAULT_OPS)
    expected = (X[:, 0] + 2.0) * X[:, 1]
    assert torch.allclose(out, expected)


def test_compute_complexity_uses_operator_weights():
    # add(mul(x0, x1), sin(x0))
    node = BinaryOpNode(
        "add",
        BinaryOpNode("mul", VariableNode(0), VariableNode(1)),
        UnaryOpNode("sin", VariableNode(0)),
    )
    cplx = compute_complexity(node, DEFAULT_OPS)
    # mul branch = 1 (mul) + 1 + 1 = 3
    # sin branch = 2 (sin) + 1 = 3
    # add root  = 1 + 3 + 3 = 7
    assert cplx == 7


def test_evaluate_tree_handles_deep_unary_chain_iteratively():
    node = VariableNode(0)
    for _ in range(1500):
        node = UnaryOpNode("neg", node)

    X = torch.tensor([[1.0], [-2.0], [3.0]], dtype=torch.float32)
    out = evaluate_tree(node, X, DEFAULT_OPS)
    assert out.shape == X[:, 0].shape
    assert torch.isfinite(out).all()
