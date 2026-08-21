"""Operator definitions for symbolic regression.

Each operator is a lightweight dataclass bundling: name, callable,
arity (1=unary, 2=binary), and a complexity score for parsimony.
All functions that can produce invalid values have safe wrappers.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Safe operator functions
# ---------------------------------------------------------------------------


def safe_div(a: Tensor, b: Tensor) -> Tensor:
    """Protected division: returns 1.0 where |b| < epsilon."""
    return torch.where(torch.abs(b) < 1e-10, torch.ones_like(a), a / b)


def safe_log(x: Tensor) -> Tensor:
    """Protected log: applies log(|x| + epsilon) to avoid log(0) / log(neg)."""
    return torch.log(torch.abs(x) + 1e-10)


def safe_exp(x: Tensor) -> Tensor:
    """Clamped exp: clamps input to [-10, 10] to prevent overflow."""
    return torch.exp(torch.clamp(x, -10.0, 10.0))


def cube(x: Tensor) -> Tensor:
    """Cube: x^3."""
    return x**3


# ---------------------------------------------------------------------------
# Operator dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operator:
    """A mathematical operator: name, callable, arity, and complexity cost."""

    name: str
    func: Callable
    arity: int
    complexity: int = 1


# ---------------------------------------------------------------------------
# Default operator sets
# ---------------------------------------------------------------------------

DEFAULT_BINARY_OPS = {
    "add": Operator("add", torch.add, arity=2, complexity=1),
    "sub": Operator("sub", torch.sub, arity=2, complexity=1),
    "mul": Operator("mul", torch.mul, arity=2, complexity=1),
    "div": Operator("div", safe_div, arity=2, complexity=2),
}

DEFAULT_UNARY_OPS = {
    "sin": Operator("sin", torch.sin, arity=1, complexity=2),
    "cos": Operator("cos", torch.cos, arity=1, complexity=2),
    "exp": Operator("exp", safe_exp, arity=1, complexity=2),
    "log": Operator("log", safe_log, arity=1, complexity=2),
    "neg": Operator("neg", torch.neg, arity=1, complexity=1),
    "square": Operator("square", torch.square, arity=1, complexity=2),
    "cube": Operator("cube", cube, arity=1, complexity=3),
}

# Binary operators that are both commutative and associative.
COMMUTATIVE_ASSOCIATIVE_BINARY_OPS = {"add", "mul"}

DEFAULT_OPS = {**DEFAULT_UNARY_OPS, **DEFAULT_BINARY_OPS}
