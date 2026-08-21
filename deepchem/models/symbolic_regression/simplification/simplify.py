"""Algebraic simplification for expression trees.

Applies rewrite rules bottom-up to remove redundant operations
(e.g., x + 0, x * 1, neg(neg(x))) and fold constant subtrees.
"""

import math
from typing import List

from deepchem.models.symbolic_regression.core.node import (
    Node,
    ConstantNode,
    UnaryOpNode,
    BinaryOpNode,
)
from deepchem.models.symbolic_regression.core.operators import (
    COMMUTATIVE_ASSOCIATIVE_BINARY_OPS,
    DEFAULT_OPS,
)

# Operator registry used for constant folding
_OPS = DEFAULT_OPS


def _is_const(node: Node, val: float) -> bool:
    """Check if a node is a ConstantNode with a specific value."""
    return isinstance(node,
                      ConstantNode) and abs(node.value.item() - val) < 1e-12


# ---------------------------------------------------------------------------
# Canonicalization helpers for commutative/associative ops
# ---------------------------------------------------------------------------


def _flatten(node: Node, op_name: str) -> List[Node]:
    """Flatten a chain like add(add(a, b), c) into [a, b, c]."""
    if isinstance(node, BinaryOpNode) and node.op_name == op_name:
        return _flatten(node.left, op_name) + _flatten(node.right, op_name)
    return [node]


def _chain(op_name: str, terms: List[Node]) -> Node:
    """Rebuild a left-associative chain from a list of terms."""
    if not terms:
        return ConstantNode(0.0 if op_name == "add" else 1.0)
    cur = terms[0]
    for term in terms[1:]:
        cur = BinaryOpNode(op_name, cur, term)
    return cur


def _canonicalize(node: Node) -> Node:
    """Sort commutative/associative chains so add(b, a) == add(a, b)."""
    if not isinstance(node, BinaryOpNode):
        return node
    if node.op_name not in COMMUTATIVE_ASSOCIATIVE_BINARY_OPS:
        return node

    terms = _flatten(node, node.op_name)
    if len(terms) <= 1:
        return node
    terms.sort(key=lambda n: str(n))
    return _chain(node.op_name, terms)


# ---------------------------------------------------------------------------
# Rewrite rules — each returns simplified node or None (no change).
# ---------------------------------------------------------------------------


def _add_zero(node: Node):
    """add(x, 0) → x, add(0, x) → x"""
    if isinstance(node, BinaryOpNode) and node.op_name == "add":
        if _is_const(node.right, 0.0):
            return node.left
        if _is_const(node.left, 0.0):
            return node.right
    return None


def _sub_zero(node: Node):
    """sub(x, 0) → x, sub(x, x) → 0"""
    if isinstance(node, BinaryOpNode) and node.op_name == "sub":
        if _is_const(node.right, 0.0):
            return node.left
        if str(node.left) == str(node.right):
            return ConstantNode(0.0)
    return None


def _mul_one(node: Node):
    """mul(x, 1) → x, mul(1, x) → x, mul(x, -1) → neg(x)"""
    if isinstance(node, BinaryOpNode) and node.op_name == "mul":
        if _is_const(node.right, 1.0):
            return node.left
        if _is_const(node.left, 1.0):
            return node.right
        if _is_const(node.right, -1.0):
            return UnaryOpNode("neg", node.left)
        if _is_const(node.left, -1.0):
            return UnaryOpNode("neg", node.right)
    return None


def _mul_zero(node: Node):
    """mul(x, 0) → 0"""
    if isinstance(node, BinaryOpNode) and node.op_name == "mul":
        if _is_const(node.right, 0.0) or _is_const(node.left, 0.0):
            return ConstantNode(0.0)
    return None


def _div_one(node: Node):
    """div(x, 1) → x, div(x, x) → 1"""
    if isinstance(node, BinaryOpNode) and node.op_name == "div":
        if _is_const(node.right, 1.0):
            return node.left
        if str(node.left) == str(node.right):
            return ConstantNode(1.0)
    return None


def _double_neg(node: Node):
    """neg(neg(x)) → x"""
    if isinstance(node, UnaryOpNode) and node.op_name == "neg":
        if isinstance(node.child, UnaryOpNode) and node.child.op_name == "neg":
            return node.child.child
    return None


def _add_neg(node: Node):
    """add(x, neg(x)) or add(neg(x), x) → 0"""
    if not isinstance(node, BinaryOpNode) or node.op_name != "add":
        return None
    left, right = node.left, node.right
    if isinstance(left, UnaryOpNode) and left.op_name == "neg":
        if str(left.child) == str(right):
            return ConstantNode(0.0)
    if isinstance(right, UnaryOpNode) and right.op_name == "neg":
        if str(right.child) == str(left):
            return ConstantNode(0.0)
    return None


def _fold_constants(node: Node):
    """Fold operations on constants: add(2, 3) → 5, sin(0) → 0, etc."""
    if isinstance(node, UnaryOpNode) and isinstance(node.child, ConstantNode):
        if node.op_name in _OPS:
            result = _OPS[node.op_name].func(node.child.value)
            value = float(result.item())
            if math.isfinite(value):
                return ConstantNode(value)
            return None

    if isinstance(node, BinaryOpNode):
        if isinstance(node.left, ConstantNode) and isinstance(
                node.right, ConstantNode):
            if node.op_name in _OPS:
                result = _OPS[node.op_name].func(node.left.value,
                                                 node.right.value)
                value = float(result.item())
                if math.isfinite(value):
                    return ConstantNode(value)
                return None

    return None


# All rules in order of application
_RULES = [
    _add_zero,
    _add_neg,
    _sub_zero,
    _mul_zero,
    _mul_one,
    _div_one,
    _double_neg,
    _fold_constants,
]

# ---------------------------------------------------------------------------
# Main simplify function
# ---------------------------------------------------------------------------


def simplify(node: Node) -> Node:
    """Simplify an expression tree by applying algebraic rewrite rules.

    Applies rules bottom-up (children first, then parent) and repeats
    until no more rules fire, handling cascading simplifications.
    """
    # Recursively simplify children first (bottom-up)
    if isinstance(node, UnaryOpNode):
        node.child = simplify(node.child)
    elif isinstance(node, BinaryOpNode):
        node.left = simplify(node.left)
        node.right = simplify(node.right)

    # Apply rules until none fire
    changed = True
    while changed:
        changed = False
        for rule in _RULES:
            result = rule(node)
            if result is not None:
                node = result
                changed = True
                break

    node = _canonicalize(node)
    return node
