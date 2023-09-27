"""
Density Functional Theory Utilities
Derived from: https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/kscalc.py
"""
try:
    import torch
except ModuleNotFoundError:
    pass

import hashlib
import xitorch as xt
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod, abstractproperty
from typing import Union, List, Dict, TypeVar, Generic, Callable, overload, Optional

from deepchem.utils.dft_utils.misc import gaussian_int

__all__ = ["CGTOBasis", "ZType"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]
# input types
AtomZsType = Union[List[str], List[ZType], torch.Tensor]
AtomPosType = Union[List[List[float]], np.ndarray, torch.Tensor]


def is_z_float(a: ZType):
    # returns if the given z-type is a floating point
    if isinstance(a, torch.Tensor):
        return a.is_floating_point()
    else:
        return isinstance(a, float)


@dataclass
class CGTOBasis:

    angmom: int
    alphas: torch.Tensor  # (nbasis,)
    coeffs: torch.Tensor  # (nbasis,)
    normalized: bool = False

    def wfnormalize_(self):
        # wavefunction normalization
        # the normalization is obtained from CINTgto_norm from
        # libcint/src/misc.c, or
        # https://github.com/sunqm/libcint/blob/b8594f1d27c3dad9034984a2a5befb9d607d4932/src/misc.c#L80

        # Please note that the square of normalized wavefunctions do not integrate
        # to 1, but e.g. for s: 4*pi, p: (4*pi/3)

        # if the basis has been normalized before, then do nothing
        if self.normalized:
            return self

        coeffs = self.coeffs

        # normalize to have individual gaussian integral to be 1 (if coeff is 1)
        coeffs = coeffs / torch.sqrt(gaussian_int(2 * self.angmom + 2, 2 * self.alphas))

        # normalize the coefficients in the basis (because some basis such as
        # def2-svp-jkfit is not normalized to have 1 in overlap)
        ee = self.alphas.unsqueeze(-1) + self.alphas.unsqueeze(-2)  # (ngauss, ngauss)
        ee = gaussian_int(2 * self.angmom + 2, ee)
        s1 = 1 / torch.sqrt(torch.einsum("a,ab,b", coeffs, ee, coeffs))
        coeffs = coeffs * s1

        self.coeffs = coeffs
        self.normalized = True
        return self


@dataclass
class AtomCGTOBasis:
    atomz: ZType
    bases: List[CGTOBasis]
    pos: torch.Tensor  # (ndim,)

# input basis type
BasisInpType = Union[str, List[CGTOBasis], List[str], List[List[CGTOBasis]],
                     Dict[Union[str, int], Union[List[CGTOBasis], str]]]

@dataclass
class DensityFitInfo:
    method: str
    auxbases: List[AtomCGTOBasis]


@dataclass
class SpinParam(Generic[T]):
    """
    Data structure to store different values for spin-up and spin-down electrons.

    Attributes
    ----------
    u: any type
        The parameters that corresponds to the spin-up electrons.
    d: any type
        The parameters that corresponds to the spin-down electrons.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/master/dqc/utils/datastruct.py
    """

    u: T
    d: T

    def sum(self):
        """
        Returns the sum of up and down parameters
        """

        return self.u + self.d

    def reduce(self, fcn: Callable) -> T:
        """
        Reduce up and down parameters with the given function
        """

        return fcn(self.u, self.d)


@dataclass
class SpinParam(Generic[T]):
    """
    Data structure to store different values for spin-up and spin-down electrons.

    Attributes
    ----------
    u: any type
        The parameters that corresponds to the spin-up electrons.
    d: any type
        The parameters that corresponds to the spin-down electrons.

    Example
    -------
    .. jupyter-execute::

        import torch
        import dqc.utils
        dens_u = torch.ones(1)
        dens_d = torch.zeros(1)
        sp = dqc.utils.SpinParam(u=dens_u, d=dens_d)
        print(sp.u)
    """
    u: T
    d: T

    def sum(a: Union[SpinParam[T], T]) -> T:
        # get the sum of up and down parameters
        if isinstance(a, SpinParam):
            return a.u + a.d  # type: ignore
        else:
            return a

    def reduce(a: Union[SpinParam[T], T], fcn: Callable[[T, T], T]) -> T:
        # reduce up and down parameters with the given function
        if isinstance(a, SpinParam):
            return fcn(a.u, a.d)
        else:
            return a

    @overload
    @staticmethod
    def apply_fcn(fcn: Callable[..., P], *a: SpinParam[T]) -> SpinParam[P]:  # type: ignore
        ...

    @overload
    @staticmethod
    def apply_fcn(fcn: Callable[..., P], *a: T) -> P:
        ...

    @staticmethod
    def apply_fcn(fcn, *a):
        # apply the function for each up and down elements of a
        assert len(a) > 0
        if isinstance(a[0], SpinParam):
            u_vals = [aa.u for aa in a]
            d_vals = [aa.d for aa in a]
            return SpinParam(u=fcn(*u_vals), d=fcn(*d_vals))
        else:
            return fcn(*a)

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
