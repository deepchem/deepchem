"""
Density Functional Theory Data Structure Utilities
"""
try:
    import torch
except ModuleNotFoundError:
    pass

from typing import Union, TypeVar

__all__ = ["ZType"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]
