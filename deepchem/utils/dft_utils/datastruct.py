"""
Density Functional Theory Data Structure Utilities
"""
from typing import Union, TypeVar, List, Dict, Generic, Callable, overload

import torch

from deepchem.utils.dft_utils.misc import gaussian_int

from dataclasses import dataclass

import numpy as np

__all__ = ["ZType"]

T = TypeVar('T')
P = TypeVar('P')

# type of the atom Z
ZType = Union[int, float, torch.Tensor]

# input types
AtomZsType = Union[List[str], List[ZType], torch.Tensor]
AtomPosType = Union[List[List[float]], np.ndarray, torch.Tensor]


def is_z_float(a: ZType):
    """Checks if the given z-type is a floating point number.]

    Parameters
    ----------
    a: ZType
        Object to check z-type of.

    Returns
    -------
    result: bool
        Whether the given z-type is a floating point number.

    """
    result = False
    if isinstance(a, torch.Tensor):
        result = a.is_floating_point()
    else:
        result = isinstance(a, float)
    return result


@dataclass
class CGTOBasis:
    """Contracted Gaussian Type Orbital (CGTO) basis
    The term contraction means "a linear combination of Gaussian
    primitives to be used as basis function." Such a basis function
    will have its coefficients and exponents fixed. The contractions
    are sometimes called Contracted Gaussian Type Orbitals.

    """

    def __init__(self,
                 angmom: int,
                 alphas: torch.Tensor,
                 coeffs: torch.Tensor,
                 normalized: bool = False):
        """Initialize this CGTO basis

        Parameters
        ----------
        angmom: int
            angular momentum of the basis
        alphas: torch.Tensor
            gaussian exponents
        coeffs: torch.Tensor
            gaussian coefficients
        normalized: bool
            whether the basis is normalized or not

        """
        self.angmom: int = angmom
        self.alphas: torch.Tensor = alphas  # (nbasis,)
        self.coeffs: torch.Tensor = coeffs  # (nbasis,)
        self.normalized: bool = normalized

    def wfnormalize_(self):
        """wavefunction normalization

        Normalization is obtained from CINTgto_norm from libcint/src/misc.c, or
        https://github.com/sunqm/libcint/blob/b8594f1d27c3dad9034984a2a5befb9d607d4932/src/misc.c#L80

        Please note that the square of normalized wavefunctions do not integrate
        to 1, but e.g. for s: 4*pi, p: (4*pi/3)

        If the basis has been normalized before, then do nothing.

        Examples
        --------
        >>> import torch
        >>> from deepchem.utils.dft_utils.datastruct import CGTOBasis
        >>> basis = CGTOBasis(1, torch.tensor([30.0]), torch.tensor([15.0]))
        >>> basis.wfnormalize_()
        CGTOBasis(angmom=1, alphas=tensor([30.]), coeffs=tensor([204.8264]), normalized=True)

        Returns
        -------
        self: CGTOBasis
            the normalized basis

        """
        if self.normalized:
            return self

        coeffs = self.coeffs

        # normalize to have individual gaussian integral to be 1 (if coeff is 1)
        coeffs = coeffs / torch.sqrt(
            gaussian_int(2 * self.angmom + 2, 2 * self.alphas))

        # normalize the coefficients in the basis (because some basis such as
        # def2-svp-jkfit is not normalized to have 1 in overlap)
        ee = self.alphas.unsqueeze(-1) + self.alphas.unsqueeze(
            -2)  # (ngauss, ngauss)
        ee = gaussian_int(2 * self.angmom + 2, ee)
        s1 = 1 / torch.sqrt(torch.einsum("a,ab,b", coeffs, ee, coeffs))
        coeffs = coeffs * s1

        self.coeffs = coeffs
        self.normalized = True
        return self

    def __repr__(self) -> str:
        """Returns a string representation of this CGTO basis.

        Returns
        -------
        angmom: int
            angular momentum of the basis
        alphas: torch.Tensor
            gaussian exponents
        coeffs: torch.Tensor
            gaussian coefficients
        normalized: bool
            whether the basis is normalized or not

        """
        return f"CGTOBasis(angmom={self.angmom}, alphas={self.alphas}, coeffs={self.coeffs}, normalized={self.normalized})"


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
        Parameters that corresponds to the spin-up electrons.
    d: any type
        Parameters that corresponds to the spin-down electrons.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/master/dqc/utils/datastruct.py

    """

    u: T
    d: T

    def __init__(self, u: T, d: T):
        self.u = u
        self.d = d

    def __repr__(self) -> str:
        """Returns a string representation of this SpinParam.

        Returns
        -------
        u: any type
            Parameters that corresponds to the spin-up electrons.
        d: any type
            Parameters that corresponds to the spin-down electrons.

        """
        return f"SpinParam(u={self.u}, d={self.d})"

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
    def apply_fcn(fcn: Callable[..., P],
                  *a: SpinParam[T]) -> SpinParam[P]:  # type: ignore
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
    grad: Optional[
        torch.
        Tensor] = None  # torch.Tensor representing (gradx, grady, gradz) with shape
    # ``(..., 3)``
    lapl: Optional[
        torch.Tensor] = None  # torch.Tensor of the laplace of the value
    kin: Optional[
        torch.Tensor] = None  # torch.Tensor of the kinetic energy density

    def __add__(self, b: ValGrad) -> ValGrad:
        return ValGrad(
            value=self.value + b.value,
            grad=self.grad + b.grad if self.grad is not None else None,
            lapl=self.lapl + b.lapl if self.lapl is not None else None,
            kin=self.kin + b.kin if self.kin is not None else None,
        )

    def __mul__(self, f: Union[float, int, torch.Tensor]) -> ValGrad:
        if isinstance(f, torch.Tensor):
            assert f.numel(
            ) == 1, "ValGrad multiplication with tensor can only be done with 1-element tensor"

        return ValGrad(
            value=self.value * f,
            grad=self.grad * f if self.grad is not None else None,
            lapl=self.lapl * f if self.lapl is not None else None,
            kin=self.kin * f if self.kin is not None else None,
        )
