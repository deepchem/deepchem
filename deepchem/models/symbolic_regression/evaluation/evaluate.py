"""Expression evaluation for symbolic regression.

Two pure functions — no state:
  evaluate_tree()   : tree + data → predictions tensor
  compute_complexity() : tree + operators → weighted complexity
"""

from typing import Dict

import torch
from torch import Tensor

from deepchem.models.symbolic_regression.core.node import (
    Node,
    ConstantNode,
    VariableNode,
    UnaryOpNode,
    BinaryOpNode,
)
from deepchem.models.symbolic_regression.core.operators import Operator


def evaluate_tree(node: Node, X: Tensor, operators: Dict[str,
                                                         Operator]) -> Tensor:
    """Evaluate an expression tree on input data.

    Evaluates with an iterative post-order traversal to avoid
    recursion-depth issues on deep trees.

    Parameters
    ----------
    node : Node
        Root of the expression tree.
    X : Tensor
        Input data of shape (n_samples, n_features).
    operators : dict[str, Operator]
        Operator registry mapping names to Operator objects.

    Returns
    -------
    Tensor
        Predictions of shape (n_samples,).
    """
    # Build post-order list iteratively.
    to_visit = [node]
    postorder = []
    while to_visit:
        current = to_visit.pop()
        postorder.append(current)
        if isinstance(current, UnaryOpNode):
            to_visit.append(current.child)
        elif isinstance(current, BinaryOpNode):
            to_visit.append(current.left)
            to_visit.append(current.right)
        elif isinstance(current, (ConstantNode, VariableNode)):
            pass
        else:
            raise TypeError(f"Unknown node type: {type(current)}")

    values: Dict[int, Tensor] = {}
    for current in reversed(postorder):
        if isinstance(current, ConstantNode):
            values[id(current)] = (
                torch.ones(X.shape[0], dtype=X.dtype, device=X.device) *
                current.value)
            continue

        if isinstance(current, VariableNode):
            values[id(current)] = X[:, current.index]
            continue

        if isinstance(current, UnaryOpNode):
            op = operators[current.op_name]
            child_val = values[id(current.child)]
            values[id(current)] = op.func(child_val)
            continue

        if isinstance(current, BinaryOpNode):
            op = operators[current.op_name]
            left_val = values[id(current.left)]
            right_val = values[id(current.right)]
            values[id(current)] = op.func(left_val, right_val)
            continue

        raise TypeError(f"Unknown node type: {type(current)}")

    return values[id(node)]


def compute_complexity(node: Node, operators: Dict[str, Operator]) -> int:
    """Compute weighted complexity of an expression tree.

    Uses the `complexity` field from each Operator instead of
    a flat node count. Leaf nodes (constant, variable) cost 1.

    Parameters
    ----------
    node : Node
        Root of the expression tree.
    operators : dict[str, Operator]
        Operator registry (provides per-operator complexity weights).

    Returns
    -------
    int
        Weighted complexity score.
    """
    total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, (ConstantNode, VariableNode)):
            total += 1
        elif isinstance(current, UnaryOpNode):
            total += operators[
                current.
                op_name].complexity if current.op_name in operators else 1
            stack.append(current.child)
        elif isinstance(current, BinaryOpNode):
            total += operators[
                current.
                op_name].complexity if current.op_name in operators else 1
            stack.append(current.left)
            stack.append(current.right)
        else:
            raise TypeError(f"Unknown node type: {type(current)}")
    return total
