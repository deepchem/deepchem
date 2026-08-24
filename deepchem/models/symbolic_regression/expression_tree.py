"""
Expression tree representation for symbolic regression.

Author: Nandini A R
Date: March 2, 2025
GSoC 2026 - DeepChem Symbolic ML
"""

import torch
from typing import Optional
from enum import Enum


class NodeType(Enum):
    """Types of nodes in expression tree."""
    OPERATOR = "operator"
    VARIABLE = "variable"
    CONSTANT = "constant"


class Operator(Enum):
    """Supported mathematical operators."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    SIN = "sin"
    COS = "cos"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    NEG = "neg"

    @property
    def arity(self) -> int:
        unary = {self.SIN, self.COS, self.EXP, self.LOG, self.SQRT, self.NEG}
        return 1 if self in unary else 2

    @property
    def symbol(self) -> str:
        symbols = {
            self.ADD: "+",
            self.SUB: "-",
            self.MUL: "*",
            self.DIV: "/",
            self.POW: "^",
            self.SIN: "sin",
            self.COS: "cos",
            self.EXP: "exp",
            self.LOG: "log",
            self.SQRT: "sqrt",
            self.NEG: "-",
        }
        return symbols[self]


class ExpressionNode:

    def __init__(
        self,
        node_type: NodeType,
        operator: Optional[Operator] = None,
        value: Optional[float] = None,
        left: Optional['ExpressionNode'] = None,
        right: Optional['ExpressionNode'] = None,
        feature_index: int = 0,
    ):
        self.node_type = node_type
        self.operator = operator
        self.value = value
        self.left = left
        self.right = right
        self.feature_index = feature_index

        if node_type == NodeType.OPERATOR:
            assert operator is not None, "Operator node must have operator"
            if operator.arity == 2:
                assert left is not None and right is not None
            elif operator.arity == 1:
                assert left is not None

        elif node_type == NodeType.CONSTANT:
            assert value is not None, "Constant node must have value"

    def is_leaf(self) -> bool:
        return self.node_type in [NodeType.VARIABLE, NodeType.CONSTANT]

    def depth(self) -> int:
        if self.is_leaf():
            return 1
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def size(self) -> int:
        if self.is_leaf():
            return 1
        left_size = self.left.size() if self.left else 0
        right_size = self.right.size() if self.right else 0
        return 1 + left_size + right_size

    def copy(self) -> 'ExpressionNode':
        if self.is_leaf():
            return ExpressionNode(
                node_type=self.node_type,
                operator=self.operator,
                value=self.value,
                feature_index=self.feature_index,
            )
        left_copy = self.left.copy() if self.left else None
        right_copy = self.right.copy() if self.right else None
        return ExpressionNode(
            node_type=self.node_type,
            operator=self.operator,
            value=self.value,
            left=left_copy,
            right=right_copy,
            feature_index=self.feature_index,
        )

    def __str__(self) -> str:
        if self.node_type == NodeType.VARIABLE:
            return f"x{self.feature_index}"

        if self.node_type == NodeType.CONSTANT:
            return f"{self.value:.3f}"

        op = self.operator
        if op.arity == 1:
            return f"{op.symbol}({self.left})"

        left_str = str(self.left)
        right_str = str(self.right)

        if op in [Operator.ADD, Operator.SUB]:
            return f"({left_str} {op.symbol} {right_str})"
        elif op in [Operator.MUL, Operator.DIV]:
            return f"{left_str} {op.symbol} {right_str}"
        elif op == Operator.POW:
            return f"{left_str}^{right_str}"
        else:
            return f"{op.symbol}({left_str}, {right_str})"


class ExpressionTree:

    def __init__(self, root: ExpressionNode):
        self.root = root

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self._evaluate_node(self.root, x)

    def _evaluate_node(self, node: ExpressionNode,
                       x: torch.Tensor) -> torch.Tensor:
        if node.node_type == NodeType.VARIABLE:
            if x.dim() == 1:
                return x  # single variable
            return x[:, node.feature_index]  # multiple features

        if node.node_type == NodeType.CONSTANT:
            if x.dim() == 1:
                return torch.full_like(x, node.value)
            return torch.full((x.shape[0], ), node.value)

        op = node.operator
        left_val = self._evaluate_node(node.left, x) if node.left else None
        right_val = self._evaluate_node(node.right, x) if node.right else None

        if op == Operator.ADD:
            return left_val + right_val
        elif op == Operator.SUB:
            return left_val - right_val
        elif op == Operator.MUL:
            return left_val * right_val
        elif op == Operator.DIV:
            return left_val / (right_val + 1e-8)
        elif op == Operator.POW:
            return torch.pow(
                torch.abs(left_val) + 1e-8, torch.clamp(right_val, -10, 10))
        elif op == Operator.SIN:
            return torch.sin(left_val)
        elif op == Operator.COS:
            return torch.cos(left_val)
        elif op == Operator.EXP:
            return torch.exp(torch.clamp(left_val, -10, 10))
        elif op == Operator.LOG:
            return torch.log(torch.abs(left_val) + 1e-8)
        elif op == Operator.SQRT:
            return torch.sqrt(torch.abs(left_val) + 1e-8)
        elif op == Operator.NEG:
            return -left_val
        else:
            raise ValueError(f"Unknown operator: {op}")

    def complexity(self) -> int:
        return self.root.size()

    def depth(self) -> int:
        return self.root.depth()

    def __str__(self) -> str:
        return f"y = {self.root}"


# Factory functions
def make_constant(value: float) -> ExpressionNode:
    return ExpressionNode(NodeType.CONSTANT, value=value)


def make_variable(feature_index: int = 0) -> ExpressionNode:
    return ExpressionNode(NodeType.VARIABLE, feature_index=feature_index)


def make_operator(op: Operator,
                  left: ExpressionNode,
                  right: Optional[ExpressionNode] = None) -> ExpressionNode:
    return ExpressionNode(
        NodeType.OPERATOR,
        operator=op,
        left=left,
        right=right,
    )
