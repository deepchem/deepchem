"""Expression tree node types for symbolic regression.

Defines the tree representation for mathematical expressions.
Four node types: ConstantNode, VariableNode, UnaryOpNode, BinaryOpNode.
"""

import copy
from abc import ABC, abstractmethod
from typing import List

import torch


class Node(ABC):
    """Base class for all expression tree nodes."""

    @abstractmethod
    def size(self) -> int:
        """Total number of nodes in this subtree."""

    @abstractmethod
    def depth(self) -> int:
        """Maximum depth of this subtree (leaf = 0)."""

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable string representation."""

    def copy(self) -> 'Node':
        """Deep copy of this subtree."""
        return copy.deepcopy(self)


class ConstantNode(Node):
    """Leaf node holding a numerical constant (stored as a scalar tensor)."""

    def __init__(self, value: float):
        self.value = torch.tensor(float(value))

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 0

    def __str__(self) -> str:
        return f"{self.value.item():.4g}"


class VariableNode(Node):
    """Leaf node referencing an input feature by index (e.g., x0, x1)."""

    def __init__(self, index: int):
        self.index = index

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 0

    def __str__(self) -> str:
        return f"x{self.index}"


class UnaryOpNode(Node):
    """Node applying a unary operator to one child (e.g., sin(x0))."""

    def __init__(self, op_name: str, child: Node):
        self.op_name = op_name
        self.child = child

    def size(self) -> int:
        return 1 + self.child.size()

    def depth(self) -> int:
        return 1 + self.child.depth()

    def __str__(self) -> str:
        return f"{self.op_name}({self.child})"


class BinaryOpNode(Node):
    """Node applying a binary operator to two children (e.g., x0 + x1)."""

    def __init__(self, op_name: str, left: Node, right: Node):
        self.op_name = op_name
        self.left = left
        self.right = right

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def __str__(self) -> str:
        return f"{self.op_name}({self.left}, {self.right})"


def get_constants(node: Node) -> List[ConstantNode]:
    """Collect all ConstantNode references from a subtree (left-to-right)."""
    if isinstance(node, ConstantNode):
        return [node]
    if isinstance(node, VariableNode):
        return []
    if isinstance(node, UnaryOpNode):
        return get_constants(node.child)
    if isinstance(node, BinaryOpNode):
        return get_constants(node.left) + get_constants(node.right)
    return []
