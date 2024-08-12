import re
import torch
import ctypes
import numpy as np
from typing import Tuple, Optional
from deepchem.utils import estimate_ovlp_rcut
from deepchem.utils.dft_utils.hamilton.intor.molintor import _gather_at_dims
from deepchem.utils.dft_utils.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CGTO
from deepchem.utils.dft_utils import LibcintWrapper, get_default_kpts, get_default_options, PBCIntOption

BLKSIZE = 128  # same as lib/gto/grid_ao_drv.c


# evaluation of the gaussian basis
def evl(shortname: str,
        wrapper: LibcintWrapper,
        rgrid: torch.Tensor,
        *,
        to_transpose: bool = False) -> torch.Tensor:
    """Evaluates the Gaussian Basis

    Examples
    --------
    >>> from deepchem.utils.dft_utils import evl, AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = evl("", wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([6, 100])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate
    to_transpose: book (default False)
        True for transposing the matrix.

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto
    """
    # expand ao_to_atom to have shape of (nao, ndim)
    ao_to_atom = wrapper.ao_to_atom().unsqueeze(-1).expand(-1, NDIM)

    # rgrid: (ngrid, ndim)
    return _EvalGTO.apply(
        # tensors
        *wrapper.params,
        rgrid,

        # nontensors or int tensors
        ao_to_atom,
        wrapper,
        shortname,
        to_transpose)


def pbc_evl(shortname: str,
            wrapper: LibcintWrapper,
            rgrid: torch.Tensor,
            kpts: Optional[torch.Tensor] = None,
            options: Optional[PBCIntOption] = None) -> torch.Tensor:
    """evaluate the basis in periodic boundary condition,
    i.e. evaluate sum_L exp(i*k*L) * phi(r - L)

    Examples
    --------
    >>> from deepchem.utils.dft_utils import pbc_evl, AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid
    >>> dtype = torch.float64
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    >>> lattice = Lattice(a)
    >>> wrap = LibcintWrapper(atombases, True, lattice)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = pbc_evl("", wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([1, 6, 100])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate. (ngrid, ndim)
    kpts: Optional[torch.Tensor] (default None)
        k-points in the Hamiltonian. (nkpts, ndim)
    options: Optional[PBCIntOption] (default None)

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto. (*ncomp, nkpts, nao, ngrid)

    """
    # get the default arguments
    kpts1 = get_default_kpts(kpts, dtype=wrapper.dtype, device=wrapper.device)
    options1 = get_default_options(options)

    # get the shifts
    coeffs, alphas, _ = wrapper.params
    rcut = estimate_ovlp_rcut(options1.precision, coeffs, alphas)
    assert wrapper.lattice is not None
    ls = wrapper.lattice.get_lattice_ls(rcut=rcut)  # (nls, ndim)

    # evaluate the gto
    exp_ikl = torch.exp(
        1j * torch.matmul(kpts1, ls.transpose(-2, -1)))  # (nkpts, nls)
    rgrid_shift = rgrid - ls.unsqueeze(-2)  # (nls, ngrid, ndim)
    ao = evl(shortname, wrapper,
             rgrid_shift.reshape(-1, NDIM))  # (*ncomp, nao, nls * ngrid)
    ao = ao.reshape(*ao.shape[:-1], ls.shape[0],
                    -1)  # (*ncomp, nao, nls, ngrid)
    out = torch.einsum("kl,...alg->...kag", exp_ikl,
                       ao.to(exp_ikl.dtype))  # (*ncomp, nkpts, nao, ngrid)
    return out


# shortcuts
def eval_gto(wrapper: LibcintWrapper,
             rgrid: torch.Tensor,
             *,
             to_transpose: bool = False) -> torch.Tensor:
    """Evaluates GTO

    Examples
    --------
    >>> from deepchem.utils.dft_utils import eval_gto, AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = eval_gto(wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([6, 100])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate
    to_transpose: book (default False)
        True for transposing the matrix.

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto

    """
    return evl("", wrapper, rgrid, to_transpose=to_transpose)


def eval_gradgto(wrapper: LibcintWrapper,
                 rgrid: torch.Tensor,
                 *,
                 to_transpose: bool = False) -> torch.Tensor:
    """Evaluates Grad GTO

    Examples
    --------
    >>> from deepchem.utils.dft_utils import eval_gradgto, AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = eval_gradgto(wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([3, 6, 100])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate
    to_transpose: book (default False)
        True for transposing the matrix.

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto
    """
    return evl("ip", wrapper, rgrid, to_transpose=to_transpose)


def eval_laplgto(wrapper: LibcintWrapper,
                 rgrid: torch.Tensor,
                 *,
                 to_transpose: bool = False) -> torch.Tensor:
    """Evaluates laplgto

    Examples
    --------
    >>> from deepchem.utils.dft_utils import eval_laplgto, AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = eval_laplgto(wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([6, 100])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate
    to_transpose: book (default False)
        True for transposing the matrix.

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto
    """
    return evl("lapl", wrapper, rgrid, to_transpose=to_transpose)


def pbc_eval_gto(wrapper: LibcintWrapper,
                 rgrid: torch.Tensor,
                 kpts: Optional[torch.Tensor] = None,
                 options: Optional[PBCIntOption] = None) -> torch.Tensor:
    """evaluate the basis in periodic boundary condition,
    i.e. evaluate sum_L exp(i*k*L) * phi(r - L)

    Examples
    --------
    >>> from deepchem.utils.dft_utils import pbc_eval_gto, AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid
    >>> dtype = torch.float64
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    >>> lattice = Lattice(a)
    >>> wrap = LibcintWrapper(atombases, True, lattice)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = pbc_eval_gto(wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([1, 6, 100])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate. (ngrid, ndim)
    kpts: Optional[torch.Tensor] (default None)
        k-points in the Hamiltonian. (nkpts, ndim)
    options: Optional[PBCIntOption] (default None)

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto. (*ncomp, nkpts, nao, ngrid)

    """
    return pbc_evl("", wrapper, rgrid, kpts, options)


def pbc_eval_gradgto(wrapper: LibcintWrapper,
                     rgrid: torch.Tensor,
                     kpts: Optional[torch.Tensor] = None,
                     options: Optional[PBCIntOption] = None) -> torch.Tensor:
    """evaluate the basis in periodic boundary condition,
    i.e. evaluate sum_L exp(i*k*L) * phi(r - L)

    Examples
    --------
    >>> from deepchem.utils.dft_utils import pbc_eval_gradgto, AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid
    >>> dtype = torch.float64
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    >>> lattice = Lattice(a)
    >>> wrap = LibcintWrapper(atombases, True, lattice)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = pbc_eval_gradgto(wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([3, 1, 6, 100])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate. (ngrid, ndim)
    kpts: Optional[torch.Tensor] (default None)
        k-points in the Hamiltonian. (nkpts, ndim)
    options: Optional[PBCIntOption] (default None)

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto. (*ncomp, nkpts, nao, ngrid)

    """
    return pbc_evl("ip", wrapper, rgrid, kpts, options)


def pbc_eval_laplgto(wrapper: LibcintWrapper,
                     rgrid: torch.Tensor,
                     kpts: Optional[torch.Tensor] = None,
                     options: Optional[PBCIntOption] = None) -> torch.Tensor:
    """evaluate the basis in periodic boundary condition,
    i.e. evaluate sum_L exp(i*k*L) * phi(r - L)

    Examples
    --------
    >>> from deepchem.utils.dft_utils import pbc_eval_laplgto, AtomCGTOBasis, LibcintWrapper, loadbasis, Lattice, RadialGrid
    >>> dtype = torch.float64
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    >>> lattice = Lattice(a)
    >>> wrap = LibcintWrapper(atombases, True, lattice)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = pbc_eval_laplgto(wrap, grid.get_rgrid())
    >>> grad.shape
    torch.Size([1, 6, 100])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    rgrid: torch.Tensor
        grid points position in the specified coordinate. (ngrid, ndim)
    kpts: Optional[torch.Tensor] (default None)
        k-points in the Hamiltonian. (nkpts, ndim)
    options: Optional[PBCIntOption] (default None)

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto. (*ncomp, nkpts, nao, ngrid)

    """
    return pbc_evl("lapl", wrapper, rgrid, kpts, options)


# pytorch function
class _EvalGTO(torch.autograd.Function):
    """wrapper class to provide the gradient for evaluating the contracted gto"""

    @staticmethod
    def forward(
            ctx,  # type: ignore
            coeffs: torch.Tensor,
            alphas: torch.Tensor,
            pos: torch.Tensor,
            rgrid: torch.Tensor,
            ao_to_atom: torch.Tensor,
            wrapper: LibcintWrapper,
            shortname: str,
            to_transpose: bool) -> torch.Tensor:
        """Forward function for EvalGTO.

        Parameters
        ----------
        coeffs: torch.Tensor
            The coefficients to get the orthogonal orbitals. (ngauss_tot)
        alphas: torch.Tensor
            gaussian exponents of the basis. (ngauss_tot)
        pos: torch.Tensor
            Position of the atom. (natom, ndim)
        rgrid: torch.Tensor
            Grid points positioned according to a center point. (ngrid, ndim)
        ao_to_atom: torch.Tensor
        (nao, ndim)
                wrapper: LibcintWrapper,
                shortname: str,
                to_transpose: bool) -> torch.Tensor:
        """

        res = gto_evaluator(wrapper, shortname, rgrid,
                            to_transpose)  # (*, nao, ngrid)
        ctx.save_for_backward(coeffs, alphas, pos, rgrid)
        ctx.other_info = (ao_to_atom, wrapper, shortname, to_transpose)
        return res

    @staticmethod
    def backward(  # type: ignore
            ctx, grad_res: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward function for EvalGTO.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object containing the saved tensors.
        grad_res : torch.Tensor
            Gradient of the result. (nao, ngrid)

        Returns
        -------
        Tuple[Optional[torch.Tensor], ...]
            Gradients for the coefficients, alphas, positions, rgrid

        """
        ao_to_atom, wrapper, shortname, to_transpose = ctx.other_info
        coeffs, alphas, pos, rgrid = ctx.saved_tensors

        if to_transpose:
            grad_res = grad_res.transpose(-2, -1)

        grad_alphas = None
        grad_coeffs = None
        if alphas.requires_grad or coeffs.requires_grad:
            u_wrapper, uao2ao = wrapper.get_uncontracted_wrapper()
            u_coeffs, u_alphas, u_pos = u_wrapper.params
            # (*, nu_ao, ngrid)
            u_grad_res = _gather_at_dims(grad_res, mapidxs=[uao2ao], dims=[-2])

            # get the scatter indices
            ao2shl = u_wrapper.ao_to_shell()

            # calculate the gradient w.r.t. coeffs
            if coeffs.requires_grad:
                grad_coeffs = torch.zeros_like(coeffs)  # (ngauss)

                # get the uncontracted version of the integral
                # (..., nu_ao, ngrid)
                dout_dcoeff = _EvalGTO.apply(*u_wrapper.params, rgrid,
                                             ao_to_atom, u_wrapper, shortname,
                                             False)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao = torch.gather(coeffs, dim=-1,
                                         index=ao2shl)  # (nu_ao)
                dout_dcoeff = dout_dcoeff / coeffs_ao[:, None]
                grad_dcoeff = torch.einsum("...ur,...ur->u", u_grad_res,
                                           dout_dcoeff)  # (nu_ao)

                grad_coeffs.scatter_add_(dim=-1, index=ao2shl, src=grad_dcoeff)

            if alphas.requires_grad:
                grad_alphas = torch.zeros_like(alphas)

                new_sname = _get_evalgto_derivname(shortname, "a")
                # (..., nu_ao, ngrid)
                dout_dalpha = _EvalGTO.apply(*u_wrapper.params, rgrid,
                                             ao_to_atom, u_wrapper, new_sname,
                                             False)

                # _alphas_ao = torch.gather(alphas, dim=-1,
                #                           index=ao2shl)  # (nu_ao)
                grad_dalpha = -torch.einsum("...ur,...ur->u", u_grad_res,
                                            dout_dalpha)

                grad_alphas.scatter_add_(dim=-1, index=ao2shl, src=grad_dalpha)

        # calculate the gradient w.r.t. basis' pos and rgrid
        grad_pos = None
        grad_rgrid = None
        if rgrid.requires_grad or pos.requires_grad:
            opsname = _get_evalgto_derivname(shortname, "r")
            dresdr = _EvalGTO.apply(*ctx.saved_tensors, ao_to_atom, wrapper,
                                    opsname, False)  # (ndim, *, nao, ngrid)
            grad_r = dresdr * grad_res  # (ndim, *, nao, ngrid)

            if rgrid.requires_grad:
                grad_rgrid = grad_r.reshape(dresdr.shape[0], -1,
                                            dresdr.shape[-1])
                grad_rgrid = grad_rgrid.sum(dim=1).transpose(
                    -2, -1)  # (ngrid, ndim)

            if pos.requires_grad:
                grad_rao = torch.movedim(grad_r, -2, 0)  # (nao, ndim, *, ngrid)
                grad_rao = -grad_rao.reshape(*grad_rao.shape[:2], -1).sum(
                    dim=-1)  # (nao, ndim)
                grad_pos = torch.zeros_like(pos)  # (natom, ndim)
                grad_pos.scatter_add_(dim=0, index=ao_to_atom, src=grad_rao)

        return grad_coeffs, grad_alphas, grad_pos, grad_rgrid, \
            None, None, None, None, None, None


# evaluator (direct interfact to libcgto)
def gto_evaluator(wrapper: LibcintWrapper, shortname: str, rgrid: torch.Tensor,
                  to_transpose: bool):
    """Evaluate the contracted GTO

    Examples
    --------
    >>> from deepchem.utils.dft_utils import gto_evaluator, AtomCGTOBasis, LibcintWrapper, loadbasis, RadialGrid
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grad = gto_evaluator(wrap, "", grid.get_rgrid(), False)
    >>> grad.shape
    torch.Size([6, 100])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    shortname: str
        Short name of the integral.
    rgrid: torch.Tensor
        grid points position in the specified coordinate (ngrid, ndim)
    to_transpose: bool
        True for transposing the matrix.

    Returns
    -------
    torch.Tensor
        Gradient for evaluating the contracted gto.
        (*, nao, ngrid) if not to_transpose else (*, ngrid, nao)

    NOTE
    ----
    This function do not propagate gradient and should only be used
    in this file only

    """
    ngrid = rgrid.shape[0]
    nshells = len(wrapper)
    nao = wrapper.nao()
    opname = _get_evalgto_opname(shortname, wrapper.spherical)
    outshape = _get_evalgto_compshape(shortname) + (nao, ngrid)

    out = np.empty(outshape, dtype=np.float64)
    non0tab = np.ones(((ngrid + BLKSIZE - 1) // BLKSIZE, nshells),
                      dtype=np.int8)
    rgrid = rgrid.contiguous()
    coords = np.asarray(rgrid, dtype=np.float64, order='F')
    ao_loc = np.asarray(wrapper.full_shell_to_aoloc, dtype=np.int32)

    c_shls = (ctypes.c_int * 2)(*wrapper.shell_idxs)
    c_ngrid = ctypes.c_int(ngrid)

    # evaluate the orbital
    operator = getattr(CGTO(), opname)
    operator.restype = ctypes.c_double
    atm, bas, env = wrapper.atm_bas_env
    operator(c_ngrid, c_shls, np2ctypes(ao_loc), np2ctypes(out),
             np2ctypes(coords), np2ctypes(non0tab), np2ctypes(atm),
             int2ctypes(atm.shape[0]), np2ctypes(bas), int2ctypes(bas.shape[0]),
             np2ctypes(env))

    if to_transpose:
        out = np.ascontiguousarray(np.moveaxis(out, -1, -2))

    out_tensor = torch.as_tensor(out,
                                 dtype=wrapper.dtype,
                                 device=wrapper.device)
    return out_tensor


def _get_evalgto_opname(shortname: str, spherical: bool) -> str:
    """Returns the complete name of the evalgto operation

    Examples
    --------
    >>> from deepchem.utils.dft_utils.hamilton.intor.gtoeval import _get_evalgto_opname
    >>> _get_evalgto_opname("", False)
    'GTOval_cart'

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    spherical : bool
        True if the basis is spherical

    Returns
    -------
    str
        Name of the evalgto operation

    """
    sname = ("_" + shortname) if (shortname != "") else ""
    suffix = "_sph" if spherical else "_cart"
    return "GTOval%s%s" % (sname, suffix)


def _get_evalgto_compshape(shortname: str) -> Tuple[int, ...]:
    """returns the component shape of the evalgto function

    Examples
    --------
    >>> from deepchem.utils.dft_utils.hamilton.intor.gtoeval import _get_evalgto_compshape
    >>> _get_evalgto_compshape("ip")
    (3,)

    Parameters
    ----------
    shortname : str
        Short name of the integral.

    Returns
    -------
    Tuple[int, ...]
        Component shape of the evalgto function

    """

    # count "ip" only at the beginning
    n_ip = len(re.findall(r"^(?:ip)*(?:ip)?", shortname)[0]) // 2
    return (NDIM,) * n_ip


def _get_evalgto_derivname(shortname: str, derivmode: str):
    """Returns the derivative name of the evalgto operation

    Examples
    --------
    >>> from deepchem.utils.dft_utils.hamilton.intor.gtoeval import _get_evalgto_derivname
    >>> _get_evalgto_derivname("", "a")
    'rr'

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    derivmode : str
        Derivative mode

    Returns
    -------
    str
        Derivative name of the evalgto operation

    """
    if derivmode == "r":
        return "ip%s" % shortname
    elif derivmode == "a":
        return "%srr" % shortname
    else:
        raise RuntimeError("Unknown derivmode: %s" % derivmode)
