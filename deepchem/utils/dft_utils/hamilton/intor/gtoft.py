from typing import Tuple, Optional
import ctypes
import torch
import numpy as np
from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, CGTOBasis
from deepchem.utils.dft_utils.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CGTO, c_null_ptr
from deepchem.utils.pytorch_utils import get_complex_dtype

__all__ = ["evl_ft", "eval_gto_ft"]

# evaluation of the Fourier Transform of the CGTO basis
def evl_ft(shortname: str, wrapper: LibcintWrapper, gvgrid: torch.Tensor) -> torch.Tensor:
    r"""
    Evaluate the Fourier Transform-ed gaussian type orbital at the given gvgrid.
    The Fourier Transform is defined as:

    $$
    F(\mathbf{G}) = \int f(\mathbf{r}) e^{-i\mathbf{G}\cdot\mathbf{r}}\ \mathrm{d}\mathbf{r}
    $$

    The results need to be divided by square root of the orbital normalization.

    Arguments
    ---------
    shortname: str
        The type of integral (currently only "" is accepted).
    wrapper: LibcintWrapper
        The gaussian basis wrapper to be evaluated.
    gvgrid: torch.Tensor
        Tensor with shape `(nggrid, ndim)` where the fourier transformed function
        is evaluated

    Returns
    -------
    torch.Tensor
        Tensor with shape `(*, nao, nggrid)` of the evaluated value. The shape
        `*` is the number of components, i.e. 3 for ``shortname == "ip"``
    """
    if shortname != "":
        raise NotImplementedError("FT evaluation for '%s' is not implemented" % shortname)
    return _EvalGTO_FT.apply(*wrapper.params, gvgrid, wrapper, shortname)

# shortcuts
def eval_gto_ft(wrapper: LibcintWrapper, gvgrid: torch.Tensor) -> torch.Tensor:
    return evl_ft("", wrapper, gvgrid)

class _EvalGTO_FT(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  # type: ignore
                # tensors not used in calculating the forward, but required
                # for the backward propagation
                alphas: torch.Tensor,  # (ngauss_tot)
                coeffs: torch.Tensor,  # (ngauss_tot)
                pos: torch.Tensor,  # (natom, ndim)

                # tensors used in forward
                gvgrid: torch.Tensor,  # (ngrid, ndim)

                # other non-tensor info
                wrapper: LibcintWrapper,
                shortname: str) -> torch.Tensor:

        res = gto_ft_evaluator(wrapper, gvgrid)  # (*, nao, ngrid)
        ctx.save_for_backward(alphas, coeffs, pos, gvgrid)
        ctx.other_info = (wrapper, shortname)
        return res

    @staticmethod
    def backward(ctx, grad_res: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        raise NotImplementedError("gradients of GTO FT evals are not implemented")

def gto_ft_evaluator(wrapper: LibcintWrapper, gvgrid: torch.Tensor) -> torch.Tensor:
    # evaluate Fourier Transform of the basis which is defined as
    # FT(f(r)) = integral(f(r) * exp(-ik.r) dr)

    # NOTE: this function do not propagate gradient and should only be used
    # in this file only
    # this is mainly from PySCF
    # https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/gto/ft_ao.py#L107

    assert gvgrid.ndim == 2
    assert gvgrid.shape[-1] == NDIM

    # gvgrid: (ngrid, ndim)
    # returns: (nao, ngrid)
    dtype = wrapper.dtype
    device = wrapper.device

    fill = CGTO().GTO_ft_fill_s1
    if wrapper.spherical:
        intor = CGTO().GTO_ft_ovlp_sph
    else:
        intor = CGTO().GTO_ft_ovlp_cart
    fn = CGTO().GTO_ft_fill_drv

    eval_gz = CGTO().GTO_Gv_general
    p_gxyzT = c_null_ptr()
    p_gs = (ctypes.c_int * 3)(0, 0, 0)
    p_b = (ctypes.c_double * 1)(0)

    # add another dummy basis to provide the multiplier
    c = np.sqrt(4 * np.pi)  # s-type normalization
    ghost_basis = CGTOBasis(
        angmom=0,
        alphas=torch.tensor([0.], dtype=dtype, device=device),
        coeffs=torch.tensor([c], dtype=dtype, device=device),
        normalized=True,
    )
    ghost_atom_basis = AtomCGTOBasis(
        atomz=0,
        bases=[ghost_basis],
        pos=torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    )
    ghost_wrapper = LibcintWrapper(
        [ghost_atom_basis], spherical=wrapper.spherical, lattice=wrapper.lattice)
    wrapper, ghost_wrapper = LibcintWrapper.concatenate(wrapper, ghost_wrapper)
    shls_slice = (*wrapper.shell_idxs, *ghost_wrapper.shell_idxs)
    ao_loc = wrapper.full_shell_to_aoloc
    atm, bas, env = wrapper.atm_bas_env

    # prepare the gvgrid
    GvT = np.asarray(gvgrid.detach().numpy().T, order="C")
    nGv = gvgrid.shape[0]

    # prepare the output matrix
    outshape = (wrapper.nao(), nGv)
    out = np.zeros(outshape, dtype=np.complex128, order="C")

    fn(intor, eval_gz, fill, np2ctypes(out),
       int2ctypes(1), (ctypes.c_int * len(shls_slice))(*shls_slice),
       np2ctypes(ao_loc),
       ctypes.c_double(0),
       np2ctypes(GvT),
       p_b, p_gxyzT, p_gs,
       int2ctypes(nGv),
       np2ctypes(atm), int2ctypes(len(atm)),
       np2ctypes(bas), int2ctypes(len(bas)),
       np2ctypes(env))

    return torch.as_tensor(out, dtype=get_complex_dtype(dtype), device=device)
