import pytest

from deepchem.models.symbolic_regression.core.node import BinaryOpNode, ConstantNode, UnaryOpNode, VariableNode
from deepchem.models.symbolic_regression.simplification.simplify import simplify

pytestmark = pytest.mark.torch


def test_simplify_identity_rules():
    x = VariableNode(0)

    out_add = simplify(BinaryOpNode("add", x.copy(), ConstantNode(0.0)))
    assert isinstance(out_add, VariableNode)
    assert out_add.index == 0

    out_mul = simplify(BinaryOpNode("mul", x.copy(), ConstantNode(1.0)))
    assert isinstance(out_mul, VariableNode)
    assert out_mul.index == 0

    out_sub = simplify(BinaryOpNode("sub", x.copy(), x.copy()))
    assert isinstance(out_sub, ConstantNode)
    assert abs(float(out_sub.value.item()) - 0.0) < 1e-12

    out_div = simplify(BinaryOpNode("div", x.copy(), x.copy()))
    assert isinstance(out_div, ConstantNode)
    assert abs(float(out_div.value.item()) - 1.0) < 1e-12


def test_simplify_double_neg_and_constant_fold():
    out_neg = simplify(UnaryOpNode("neg", UnaryOpNode("neg", VariableNode(1))))
    assert isinstance(out_neg, VariableNode)
    assert out_neg.index == 1

    out_fold = simplify(
        BinaryOpNode("add", ConstantNode(2.0), ConstantNode(3.0)))
    assert isinstance(out_fold, ConstantNode)
    assert abs(float(out_fold.value.item()) - 5.0) < 1e-12


def test_simplify_canonicalizes_commutative_associative_forms():
    node = BinaryOpNode(
        "add",
        VariableNode(2),
        BinaryOpNode("add", VariableNode(0), VariableNode(1)),
    )
    out = simplify(node)
    # Canonical sorted + left-associated form.
    assert str(out) == "add(add(x0, x1), x2)"
