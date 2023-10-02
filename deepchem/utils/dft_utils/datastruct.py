"""
Density Functional Theory Data Structure Utilities
"""
from typing import Union, TypeVar

import torch

__all__ = ["ZType"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]
