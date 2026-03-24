import torch
import numpy as np
from typing import Tuple, Optional
from deepchem.utils import get_complex_dtype
from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, CGTOBasis
from deepchem.utils.analytical_integrators.integrals import gto_ft_evaluator_py


NDIM = 3


def evl_ft(shortname: str, wrapper: LibcintWrapper,
           gvgrid: torch.Tensor) -> torch.Tensor:
    r"""
    Evaluate the Fourier Transform-ed gaussian type orbital at the given gvgrid.
    The Fourier Transform is defined as:

    $$
    F(\mathbf{G}) = \int f(\mathbf{r}) e^{-i\mathbf{G}\cdot\mathbf{r}}\ \mathrm{d}\mathbf{r}
    $$

    The results need to be divided by square root of the orbital normalization.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, CGTOBasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.gtoft import evl_ft
    >>> # Create a simple basis
    >>> basis = CGTOBasis(angmom=0, alphas=torch.tensor([1.0]),
    ...                   coeffs=torch.tensor([1.0]), normalized=True)
    >>> atom = AtomCGTOBasis(atomz=1, bases=[basis],
    ...                      pos=torch.tensor([0.0, 0.0, 0.0]))
    >>> wrapper = LibcintWrapper([atom])
    >>> # Create grid points
    >>> gvgrid = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> result = evl_ft("", wrapper, gvgrid)

    Parameters
    ----------
    shortname: str
        Type of integral (currently only "" is accepted).
    wrapper: LibcintWrapper
        Gaussian basis wrapper to be evaluated.
    gvgrid: torch.Tensor
        Tensor with shape `(nggrid, ndim)` where the fourier transformed function
        is evaluated.

    Returns
    -------
    torch.Tensor
        Tensor with shape `(*, nao, nggrid)` of the evaluated value. The shape
        `*` is the number of components, i.e. 3 for ``shortname == "ip"``

    """
    if shortname != "":
        raise NotImplementedError("FT evaluation for '%s' is not implemented" %
                                  shortname)
    return _EvalGTO_FT.apply(*wrapper.params, gvgrid, wrapper, shortname)


# shortcuts


def eval_gto_ft(wrapper: LibcintWrapper, gvgrid: torch.Tensor) -> torch.Tensor:
    r"""
    Evaluate the Fourier Transform of Gaussian type orbitals at the given gvgrid.

    The Fourier Transform is defined as:

    $$
    F(\mathbf{G}) = \int f(\mathbf{r}) e^{-i\mathbf{G}\cdot\mathbf{r}}\ \mathrm{d}\mathbf{r}
    $$

    The results need to be divided by square root of the orbital normalization.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, CGTOBasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.gtoft import eval_gto_ft
    >>> # Create a simple basis
    >>> basis = CGTOBasis(angmom=0, alphas=torch.tensor([1.0]),
    ...                   coeffs=torch.tensor([1.0]), normalized=True)
    >>> atom = AtomCGTOBasis(atomz=1, bases=[basis],
    ...                      pos=torch.tensor([0.0, 0.0, 0.0]))
    >>> wrapper = LibcintWrapper([atom])
    >>> # Create grid points in reciprocal space
    >>> gvgrid = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> result = eval_gto_ft(wrapper, gvgrid)
    >>> result.shape
    torch.Size([1, 2])

    Parameters
    ----------
    wrapper: LibcintWrapper
        Gaussian basis wrapper to be evaluated.
    gvgrid: torch.Tensor
        Tensor with shape `(nggrid, ndim)` where the fourier transformed function
        is evaluated.

    Returns
    -------
    torch.Tensor
        Tensor with shape `(nao, nggrid)` of the evaluated value.

    """
    return evl_ft("", wrapper, gvgrid)


class _EvalGTO_FT(torch.autograd.Function):
    r"""
    Autograd function for evaluating the Fourier Transform of Gaussian type orbitals.

    This class implements the forward pass for computing the Fourier transform
    of Gaussian-type orbitals. The Fourier Transform is defined as:

    $$
    F(\mathbf{G}) = \int f(\mathbf{r}) e^{-i\mathbf{G}\cdot\mathbf{r}}\ \mathrm{d}\mathbf{r}
    $$

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, CGTOBasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.gtoft import _EvalGTO_FT
    >>> # Create a simple hydrogen atom basis
    >>> basis = CGTOBasis(angmom=0, alphas=torch.tensor([1.0]),
    ...                   coeffs=torch.tensor([1.0]), normalized=True)
    >>> atom = AtomCGTOBasis(atomz=1, bases=[basis],
    ...                      pos=torch.tensor([0.0, 0.0, 0.0]))
    >>> wrapper = LibcintWrapper([atom])
    >>> # Create grid points in reciprocal space
    >>> gvgrid = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> # Get parameters from wrapper
    >>> alphas = wrapper.params[0]
    >>> coeffs = wrapper.params[1]
    >>> pos = wrapper.params[2]
    >>> result = _EvalGTO_FT.apply(alphas, coeffs, pos, gvgrid, wrapper, "")
    >>> result.shape
    torch.Size([1, 2])

    Note
    ----
    This is an internal class. Use :func:`eval_gto_ft` for evaluating GTO Fourier transforms.

    """

    @staticmethod
    def forward(
            ctx,  # type: ignore
            alphas: torch.Tensor,
            coeffs: torch.Tensor,
            pos: torch.Tensor,
            gvgrid: torch.Tensor,
            wrapper: LibcintWrapper,
            shortname: str) -> torch.Tensor:
        """Forward pass of _EvalGTO_FT.

        Parameters
        ----------
        alphas: torch.Tensor
            Gaussian exponent parameters with shape (ngauss_tot,)
        coeffs: torch.Tensor
            Gaussian coefficient parameters with shape (ngauss_tot,)
        pos: torch.Tensor
            Atomic positions with shape (natom, ndim)
        gvgrid: torch.Tensor
            Grid points for Fourier transform evaluation with shape (ngrid, ndim)
        wrapper: LibcintWrapper
            Integral wrapper containing basis function information
        shortname: str
            Name of the integral type (currently only empty string is supported)

        Returns
        -------
        torch.Tensor
            Fourier-transformed orbital values with shape (*, nao, ngrid)

        """
        res = gto_ft_evaluator(wrapper, gvgrid)  # (*, nao, ngrid)
        ctx.save_for_backward(alphas, coeffs, pos, gvgrid)
        ctx.other_info = (wrapper, shortname)
        return res

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Backward pass of _EvalGTO_FT.

        Parameters
        ----------
        *grad_outputs: torch.Tensor
            Gradients of the loss with respect to the output tensor

        Returns
        -------
        Tuple[Optional[torch.tensor], ...]
            Gradients with respect to input parameters (alphas, coeffs, pos, gvgrid, wrapper, shortname)

        Raises
        ------
        NotImplementedError
            Gradients for GTO Fourier transform evaluation are not implemented

        """
        raise NotImplementedError(
            "Gradients of GTO FT evals are not implemented")


def gto_ft_evaluator(wrapper: LibcintWrapper,
                     gvgrid: torch.Tensor) -> torch.Tensor:
    """Evaluates Fourier Transform of the Gaussian type orbital basis functions.

    The Fourier Transform is defined as:
    FT(f(r)) = ∫f(r) * exp(-ik·r) dr

    NOTE: This function does not propagate gradients and should only be used
    internally within this module. The implementation is primarily based on PySCF.
    https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/gto/ft_ao.py#L107

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, CGTOBasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.gtoft import gto_ft_evaluator
    >>> # Create a simple hydrogen atom basis
    >>> basis = CGTOBasis(angmom=0, alphas=torch.tensor([1.0]),
    ...                   coeffs=torch.tensor([1.0]), normalized=True)
    >>> atom = AtomCGTOBasis(atomz=1, bases=[basis],
    ...                      pos=torch.tensor([0.0, 0.0, 0.0]))
    >>> wrapper = LibcintWrapper([atom])
    >>> # Create grid points in reciprocal space
    >>> gvgrid = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> result = gto_ft_evaluator(wrapper, gvgrid)
    >>> result.shape
    torch.Size([1, 2])

    Parameters
    ----------
    wrapper: LibcintWrapper
        Wrapper containing the Gaussian basis functions to be transformed
    gvgrid: torch.Tensor
        Grid points in reciprocal space with shape (ngrid, 3) where the
        Fourier transform is evaluated

    Returns
    -------
    torch.Tensor
        Fourier-transformed orbital values with shape (nao, ngrid)
        where nao is the number of atomic orbitals and ngrid is the number
        of grid points

    """

    assert gvgrid.ndim == 2
    assert gvgrid.shape[-1] == NDIM

    # gvgrid: (ngrid, ndim)
    # returns: (nao, ngrid)
    dtype = wrapper.dtype
    device = wrapper.device

    out = gto_ft_evaluator_py(wrapper, gvgrid)

    return torch.as_tensor(out, dtype=get_complex_dtype(dtype), device=device)
