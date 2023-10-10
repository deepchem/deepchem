"""
Density Functional Theory Data Structure Utilities
"""
try:
    import torch
except ModuleNotFoundError:
    pass

from dataclasses import dataclass
from typing import Union, TypeVar, Generic, Callable, Optional

__all__ = ["ZType"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]

T = TypeVar('T')


@dataclass
class SpinParam(Generic[T]):
    """
    Data structure to store different values for spin-up and spin-down electrons.

    Examples
    --------
    >>> import torch
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
    """
    Data structure that contains local information about density profiles.

    Attributes
    ----------
    value: torch.Tensor
        Tensors containing the value of the local information.
    grad: torch.Tensor or None
        If tensor, it represents the gradient of the local information with shape
        ``(..., 3)`` where ``...`` should be the same shape as ``value``.
    lapl: torch.Tensor or None
        If tensor, represents the laplacian value of the local information.
        It should have the same shape as ``value``.
    kin: torch.Tensor or None
        If tensor, represents the local kinetic energy density.
        It should have the same shape as ``value``.
    """
    # data structure used as a umbrella class for density profiles and
    # the derivative of the potential w.r.t. density profiles

    value: torch.Tensor  # torch.Tensor of the value in the grid
    grad: Optional[torch.Tensor] = None  # torch.Tensor representing (gradx, grady, gradz) with shape
    # ``(..., 3)``
    lapl: Optional[torch.Tensor] = None  # torch.Tensor of the laplace of the value
    kin: Optional[torch.Tensor] = None  # torch.Tensor of the kinetic energy density

    def __add__(self, b: ValGrad) -> ValGrad:
        return ValGrad(
            value=self.value + b.value,
            grad=self.grad + b.grad if self.grad is not None else None,
            lapl=self.lapl + b.lapl if self.lapl is not None else None,
            kin=self.kin + b.kin if self.kin is not None else None,
        )

    def __mul__(self, f: Union[float, int, torch.Tensor]) -> ValGrad:
        if isinstance(f, torch.Tensor):
            assert f.numel() == 1, "ValGrad multiplication with tensor can only be done with 1-element tensor"

        return ValGrad(
            value=self.value * f,
            grad=self.grad * f if self.grad is not None else None,
            lapl=self.lapl * f if self.lapl is not None else None,
            kin=self.kin * f if self.kin is not None else None,
        )