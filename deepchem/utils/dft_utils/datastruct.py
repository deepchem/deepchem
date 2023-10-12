"""
Density Functional Theory Data Structure Utilities
"""
import torch

from dataclasses import dataclass
from typing import Union, TypeVar, Generic, Callable, Optional

__all__ = ["ZType", "SpinParam", "ValGrad"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]


@dataclass
class SpinParam(Generic[T]):
    """
    Data structure to store different values for spin-up and spin-down electrons.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.datastruct import SpinParam
    >>> sp = SpinParam(1, 2)
    >>> sp.u
    1
    >>> sp.d
    2
    >>> sp.sum()
    3
    >>> sp.reduce(lambda x, y: x * y)
    2

    References
    ----------
    .. [1] Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation\
    functional from nature with fully differentiable density functional\
    theory." Physical Review Letters 127.12 (2021): 126403.\
    https://github.com/diffqc/dqc/blob/master/dqc/utils/datastruct.py

    """

    def __init__(self, u: T, d: T):
        """Initialize SpinParam

        Parameters
        ----------
        u: any type
            Parameters that corresponds to the spin-up electrons.
        d: any type
            Parameters that corresponds to the spin-down electrons.

        """
        self.u = u
        self.d = d

    def sum(self):
        """Returns the sum of up and down parameters"""
        return self.u + self.d

    def reduce(self, fcn: Callable) -> T:
        """Reduce up and down parameters with the given function"""
        return fcn(self.u, self.d)


@dataclass
class ValGrad:
    """Data structure that contains local information about density profiles.
    data structure used as a umbrella class for density profiles and the
    derivative of the potential w.r.t. density profiles.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.datastruct import ValGrad
    >>> vg1 = ValGrad(torch.tensor([1, 2, 3]))
    >>> vg1.value
    tensor([1, 2, 3])
    >>> vg1.grad = torch.tensor([4, 5, 6])
    >>> vg1.grad
    tensor([4, 5, 6])
    >>> vg2 = ValGrad(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
    >>> vg1 + vg2
    ValGrad(value=tensor([2, 4, 6]), grad=tensor([ 8, 10, 12]), lapl=None, kin=None)

    """

    def __init__(self,
                 value: torch.Tensor,
                 grad: Optional[torch.Tensor] = None,
                 lapl: Optional[torch.Tensor] = None,
                 kin: Optional[torch.Tensor] = None):
        """Initialize ValGrad

        Parameters
        ----------
        value: torch.Tensor
            Tensors containing the value of the local information.
        grad: torch.Tensor or None
            If tensor, it represents the gradient of the local information with shape
            ``(_, 3)`` where `...` should be the same shape as `value`.
            (gradx, grady, gradz)
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

    def __repr__(self):
        """Representation of ValGrad object

        Returns
        -------
        value: torch.Tensor
            Tensors containing the value of the local information.
        grad: torch.Tensor or None
            If tensor, it represents the gradient of the local information with shape
            ``(_, 3)`` where `...` should be the same shape as `value`.
            (gradx, grady, gradz)
        lapl: torch.Tensor or None
            If tensor, represents the laplacian value of the local information.
            It should have the same shape as ``value``.
        kin: torch.Tensor or None
            If tensor, represents the local kinetic energy density.
            It should have the same shape as ``value``.

        """
        return f"ValGrad(value={self.value}, grad={self.grad}, lapl={self.lapl}, kin={self.kin})"

    def __add__(self, b):
        """Add two ValGrad objects together

        Parameters
        ----------
        b: ValGrad
            The other ValGrad object to be added

        Returns
        -------
        c: ValGrad
            The sum of the two ValGrad objects

        """
        c: ValGrad = ValGrad(
            value=self.value + b.value,
            grad=self.grad + b.grad if self.grad is not None else None,
            lapl=self.lapl + b.lapl if self.lapl is not None else None,
            kin=self.kin + b.kin if self.kin is not None else None,
        )
        return c

    def __mul__(self, f: Union[float, int, torch.Tensor]):
        if isinstance(f, torch.Tensor):
            assert f.numel(
            ) == 1, "ValGrad multiplication with tensor can only be done with 1-element tensor."

        return ValGrad(
            value=self.value * f,
            grad=self.grad * f if self.grad is not None else None,
            lapl=self.lapl * f if self.lapl is not None else None,
            kin=self.kin * f if self.kin is not None else None,
        )
