"""
Density Functional Theory Data Structure Utilities
"""
from typing import Union, TypeVar, Generic, Optional, Callable, List
from dataclasses import dataclass
import torch
import numpy as np

__all__ = ["ZType"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]

# input types
AtomZsType = Union[List[str], List[ZType], torch.Tensor]
AtomPosType = Union[List[List[float]], np.ndarray, torch.Tensor]


@dataclass
class SpinParam(Generic[T]):
    """Data structure to store different values for spin-up and spin-down electrons.
    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import SpinParam
    >>> dens_u = torch.ones(1)
    >>> dens_d = torch.zeros(1)
    >>> sp = SpinParam(u=dens_u, d=dens_d)
    >>> sp.u
    tensor([1.])
    >>> sp.sum()
    tensor([1.])
    >>> sp.reduce(torch.multiply)
    tensor([0.])
    """

    def __init__(self, u: T, d: T):
        """Initialize the SpinParam object.
        Parameters
        ----------
        u: any type
            The parameters that corresponds to the spin-up electrons.
        d: any type
            The parameters that corresponds to the spin-down electrons.
        """
        self.u = u
        self.d = d

    def __repr__(self) -> str:
        """Return the string representation of the SpinParam object."""
        return f"SpinParam(u={self.u}, d={self.d})"

    def sum(self):
        """Returns the sum of up and down parameters."""

        return self.u + self.d

    def reduce(self, fcn: Callable) -> T:
        """Reduce up and down parameters with the given function."""

        return fcn(self.u, self.d)


@dataclass
class ValGrad:
    """Data structure that contains local information about density profiles.
    Data structure used as a umbrella class for density profiles and the
    derivative of the potential w.r.t. density profiles.
    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import ValGrad
    >>> dens = torch.ones(1)
    >>> grad = torch.zeros(1)
    >>> lapl = torch.ones(1)
    >>> kin = torch.ones(1)
    >>> vg = ValGrad(value=dens, grad=grad, lapl=lapl, kin=kin)
    >>> vg + vg
    ValGrad(value=tensor([2.]), grad=tensor([0.]), lapl=tensor([2.]), kin=tensor([2.]))
    >>> vg * 5
    ValGrad(value=tensor([5.]), grad=tensor([0.]), lapl=tensor([5.]), kin=tensor([5.]))
    """

    def __init__(self,
                 value: torch.Tensor,
                 grad: Optional[torch.Tensor] = None,
                 lapl: Optional[torch.Tensor] = None,
                 kin: Optional[torch.Tensor] = None):
        """Initialize the ValGrad object.
        Parameters
        ----------
        value: torch.Tensor
            Tensors containing the value of the local information.
        grad: torch.Tensor or None
            If tensor, it represents the gradient of the local information with
            shape ``(..., 3)`` where ``...`` should be the same shape as ``value``.
        lapl: torch.Tensor or None
            If tensor, represents the laplacian value of the local information.
            It should have the same shape as ``value``.
        kin: torch.Tensor or None
            If tensor, represents the local kinetic energy density.
            It should have the same shape as ``value``.
        """
        self.value = value
        self.grad = grad
        self.lapl = lapl
        self.kin = kin

    def __add__(self, b):
        """Add two ValGrad objects together."""
        return ValGrad(
            value=self.value + b.value,
            grad=self.grad + b.grad if self.grad is not None else None,
            lapl=self.lapl + b.lapl if self.lapl is not None else None,
            kin=self.kin + b.kin if self.kin is not None else None,
        )

    def __mul__(self, f: Union[float, int, torch.Tensor]):
        """Multiply the ValGrad object with a scalar."""
        if isinstance(f, torch.Tensor):
            assert f.numel(
            ) == 1, "ValGrad multiplication with tensor can only be done with 1-element tensor"

        return ValGrad(
            value=self.value * f,
            grad=self.grad * f if self.grad is not None else None,
            lapl=self.lapl * f if self.lapl is not None else None,
            kin=self.kin * f if self.kin is not None else None,
        )

    def __repr__(self):
        return f"ValGrad(value={self.value}, grad={self.grad}, lapl={self.lapl}, kin={self.kin})"
