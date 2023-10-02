"""
Density Functional Theory Data Structure Utilities
"""
try:
    import torch
except ModuleNotFoundError:
    pass

from typing import Union, TypeVar, List
from dataclasses import dataclass
import numpy as np

from deepchem.utils.dft_utils.misc import gaussian_int

__all__ = ["ZType"]

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
    """Contracted Gaussian Type Orbital (CGTO) basis

    Defin
    -----
    CGTO basis is defined as a linear combination of gaussian functions.

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
            gaussian exponents (nbasis,)
        coeffs: torch.Tensor
            gaussian coefficients (nbasis,)
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
        >>> basis = CGTOBasis(0, torch.tensor([1.0]), torch.tensor([1.0]))
        >>> basis.wfnormalize_()
        CGTOBasis(angmom=0, alphas=tensor([1.]), coeffs=tensor([1.]), normalized=True)

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
