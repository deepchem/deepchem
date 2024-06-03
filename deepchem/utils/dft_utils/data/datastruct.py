"""
Density Functional Theory Data Structure Utilities
"""
from typing import Any, Union, TypeVar, Generic, Optional, Callable, List, Dict
from dataclasses import dataclass
import torch
import numpy as np
from deepchem.utils import gaussian_integral

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

    def sum(a: Union['SpinParam[T]', T]) -> Any:
        """Returns the sum of up and down parameters."""
        if isinstance(a, SpinParam):
            return a.u + a.d  # type: ignore
        else:
            return a

    def reduce(a: Union['SpinParam[T]', T], fcn: Callable[[T, T], T]) -> T:
        """Reduce up and down parameters with the given function."""
        if isinstance(a, SpinParam):
            return fcn(a.u, a.d)
        else:
            return a

    @staticmethod
    def apply_fcn(fcn: Callable[..., P], *a):
        """"Apply the function for each up and down elements of a"""
        assert len(a) > 0
        if isinstance(a[0], SpinParam):
            u_vals = [aa.u for aa in a]
            d_vals = [aa.d for aa in a]
            return SpinParam(u=fcn(*u_vals), d=fcn(*d_vals))
        else:
            return fcn(*a)


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


@dataclass
class CGTOBasis:
    """Data structure that contains information about a contracted gaussian
    type orbital (CGTO).

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import CGTOBasis
    >>> alphas = torch.ones(1)
    >>> coeffs = torch.ones(1)
    >>> cgto = CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs)
    >>> cgto.wfnormalize_()
    CGTOBasis(angmom=0, alphas=tensor([1.]), coeffs=tensor([2.5265]), normalized=True)

    """

    def __init__(self,
                 angmom: int,
                 alphas: torch.Tensor,
                 coeffs: torch.Tensor,
                 normalized: bool = False):
        """Initialize the CGTOBasis object.

        Parameters
        ----------
        angmom: int
            The angular momentum of the basis.
        alphas: torch.Tensor
            The gaussian exponents of the basis. Shape: (nbasis,)
        coeffs: torch.Tensor
            The coefficients of the basis. Shape: (nbasis,)

        """
        self.angmom = angmom
        self.alphas = alphas
        self.coeffs = coeffs
        self.normalized = normalized

    def __repr__(self):
        """Return the string representation of the CGTOBasis object.

        Returns
        -------
        angmom: int
            The angular momentum of the basis.
        alphas: torch.Tensor
            The gaussian exponents of the basis. Shape: (nbasis,)
        coeffs: torch.Tensor
            The coefficients of the basis. Shape: (nbasis,)

        """
        return f"CGTOBasis(angmom={self.angmom}, alphas={self.alphas}, coeffs={self.coeffs}, normalized={self.normalized})"

    def wfnormalize_(self) -> "CGTOBasis":
        """Wavefunction normalization

        The normalization is obtained from CINTgto_norm from
        libcint/src/misc.c, or
        https://github.com/sunqm/libcint/blob/b8594f1d27c3dad9034984a2a5befb9d607d4932/src/misc.c#L80

        Please note that the square of normalized wavefunctions do not integrate
        to 1, but e.g. for s: 4*pi, p: (4*pi/3)

        """

        # if the basis has been normalized before, then do nothing
        if self.normalized:
            return self

        coeffs = self.coeffs

        # normalize to have individual gaussian integral to be 1 (if coeff is 1)
        value = gaussian_integral(2 * self.angmom + 2, 2 * self.alphas)
        assert isinstance(value, torch.Tensor)
        coeffs = coeffs / torch.sqrt(value)

        # normalize the coefficients in the basis (because some basis such as
        # def2-svp-jkfit is not normalized to have 1 in overlap)
        ee = self.alphas.unsqueeze(-1) + self.alphas.unsqueeze(
            -2)  # (ngauss, ngauss)
        ee = gaussian_integral(2 * self.angmom + 2, ee)  # type: ignore
        s1 = 1 / torch.sqrt(torch.einsum("a,ab,b", coeffs, ee, coeffs))
        coeffs = coeffs * s1

        self.coeffs = coeffs
        self.normalized = True
        return self


@dataclass
class AtomCGTOBasis:
    """Data structure that contains information about a atom and its contracted
    gaussian type orbital (CGTO).

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, CGTOBasis
    >>> alphas = torch.ones(1)
    >>> coeffs = torch.ones(1)
    >>> cgto = CGTOBasis(angmom=0, alphas=alphas, coeffs=coeffs)
    >>> atomcgto = AtomCGTOBasis(atomz=1, bases=[cgto], pos=[[0.0, 0.0, 0.0]])
    >>> atomcgto
    AtomCGTOBasis(atomz=1, bases=[CGTOBasis(angmom=0, alphas=tensor([1.]), coeffs=tensor([1.]), normalized=False)], pos=tensor([[0., 0., 0.]]))

    """

    def __init__(self, atomz: ZType, bases: List[CGTOBasis], pos: AtomPosType):
        """Initialize the AtomCGTOBasis object.

        Parameters
        ----------
        atomz: ZType
            Atomic number of the atom.
        bases: List[CGTOBasis]
            List of CGTOBasis objects.
        pos: AtomPosType
            Position of the atom. Shape: (ndim,)

        """
        self.atomz = atomz
        self.bases = bases
        if isinstance(pos, torch.Tensor):
            self.pos = pos
        else:
            self.pos = torch.tensor(pos)

    def __repr__(self):
        """Return the string representation of the AtomCGTOBasis object.

        Returns
        -------
        atomz: ZType
            Atomic number of the atom.
        bases: List[CGTOBasis]
            List of CGTOBasis objects.
        pos: AtomPosType
            Position of the atom.

        """
        return f"AtomCGTOBasis(atomz={self.atomz}, bases={self.bases}, pos={self.pos})"


# input basis type
BasisInpType = Union[str, List[CGTOBasis], List[str], List[List[CGTOBasis]],
                     Dict[Union[str, int], Union[List[CGTOBasis], str]]]
