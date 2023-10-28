"""
Density Functional Theory Data Structure Utilities
"""
from typing import Union, TypeVar, List
import numpy as np
import torch

__all__ = ["ZType"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]
# input types
AtomZsType = Union[List[str], List[ZType], torch.Tensor]
AtomPosType = Union[List[List[float]], np.ndarray, torch.Tensor]
