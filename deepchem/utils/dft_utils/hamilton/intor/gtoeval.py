import re
import ctypes
from typing import Tuple, Optional
import torch
import numpy as np
from dqc.hamilton.intor.lcintwrap import LibcintWrapper
from dqc.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CGTO
from dqc.hamilton.intor.pbcintor import _get_default_kpts, _get_default_options, PBCIntOption
from dqc.utils.pbc import estimate_ovlp_rcut
from dqc.hamilton.intor.molintor import _gather_at_dims

__all__ = ["evl", "eval_gto", "eval_gradgto", "eval_laplgto",
           "pbc_evl", "pbc_eval_gto", "pbc_eval_gradgto", "pbc_eval_laplgto"]

BLKSIZE = 128  # same as lib/gto/grid_ao_drv.c

# evaluation of the gaussian basis
def evl(shortname: str, wrapper: LibcintWrapper, rgrid: torch.Tensor,
        *, to_transpose: bool = False) -> torch.Tensor:
    # expand ao_to_atom to have shape of (nao, ndim)
    ao_to_atom = wrapper.ao_to_atom().unsqueeze(-1).expand(-1, NDIM)

    # rgrid: (ngrid, ndim)
    return _EvalGTO.apply(
        # tensors
        *wrapper.params, rgrid,

        # nontensors or int tensors
        ao_to_atom, wrapper, shortname, to_transpose)

def pbc_evl(shortname: str, wrapper: LibcintWrapper, rgrid: torch.Tensor,
            kpts: Optional[torch.Tensor] = None,
            options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # evaluate the basis in periodic boundary condition, i.e. evaluate
    # sum_L exp(i*k*L) * phi(r - L)
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # ls: (nls, ndim)
    # returns: (*ncomp, nkpts, nao, ngrid)

    # get the default arguments
    kpts1 = _get_default_kpts(kpts, dtype=wrapper.dtype, device=wrapper.device)
    options1 = _get_default_options(options)

    # get the shifts
    coeffs, alphas, _ = wrapper.params
    rcut = estimate_ovlp_rcut(options1.precision, coeffs, alphas)
    assert wrapper.lattice is not None
    ls = wrapper.lattice.get_lattice_ls(rcut=rcut)  # (nls, ndim)

    # evaluate the gto
    exp_ikl = torch.exp(1j * torch.matmul(kpts1, ls.transpose(-2, -1)))  # (nkpts, nls)
    rgrid_shift = rgrid - ls.unsqueeze(-2)  # (nls, ngrid, ndim)
    ao = evl(shortname, wrapper, rgrid_shift.reshape(-1, NDIM))  # (*ncomp, nao, nls * ngrid)
    ao = ao.reshape(*ao.shape[:-1], ls.shape[0], -1)  # (*ncomp, nao, nls, ngrid)
    out = torch.einsum("kl,...alg->...kag", exp_ikl, ao.to(exp_ikl.dtype))  # (*ncomp, nkpts, nao, ngrid)
    return out

# shortcuts
def eval_gto(wrapper: LibcintWrapper, rgrid: torch.Tensor, *, to_transpose: bool = False) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (nao, ngrid)
    return evl("", wrapper, rgrid, to_transpose=to_transpose)

def eval_gradgto(wrapper: LibcintWrapper, rgrid: torch.Tensor, *, to_transpose: bool = False) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (ndim, nao, ngrid)
    return evl("ip", wrapper, rgrid, to_transpose=to_transpose)

def eval_laplgto(wrapper: LibcintWrapper, rgrid: torch.Tensor, *, to_transpose: bool = False) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (nao, ngrid)
    return evl("lapl", wrapper, rgrid, to_transpose=to_transpose)

def pbc_eval_gto(wrapper: LibcintWrapper, rgrid: torch.Tensor,
                 kpts: Optional[torch.Tensor] = None,
                 options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # return: (nkpts, nao, ngrid)
    return pbc_evl("", wrapper, rgrid, kpts, options)

def pbc_eval_gradgto(wrapper: LibcintWrapper, rgrid: torch.Tensor,
                     kpts: Optional[torch.Tensor] = None,
                     options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # return: (ndim, nkpts, nao, ngrid)
    return pbc_evl("ip", wrapper, rgrid, kpts, options)

def pbc_eval_laplgto(wrapper: LibcintWrapper, rgrid: torch.Tensor,
                     kpts: Optional[torch.Tensor] = None,
                     options: Optional[PBCIntOption] = None) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # kpts: (nkpts, ndim)
    # return: (nkpts, nao, ngrid)
    return pbc_evl("lapl", wrapper, rgrid, kpts, options)

################## pytorch function ##################
class _EvalGTO(torch.autograd.Function):
    # wrapper class to provide the gradient for evaluating the contracted gto
    @staticmethod
    def forward(ctx,  # type: ignore
                # tensors not used in calculating the forward, but required
                # for the backward propagation
                coeffs: torch.Tensor,  # (ngauss_tot)
                alphas: torch.Tensor,  # (ngauss_tot)
                pos: torch.Tensor,  # (natom, ndim)

                # tensors used in forward
                rgrid: torch.Tensor,  # (ngrid, ndim)

                # other non-tensor info
                ao_to_atom: torch.Tensor,  # int tensor (nao, ndim)
                wrapper: LibcintWrapper,
                shortname: str,
                to_transpose: bool) -> torch.Tensor:

        res = gto_evaluator(wrapper, shortname, rgrid, to_transpose)  # (*, nao, ngrid)
        ctx.save_for_backward(coeffs, alphas, pos, rgrid)
        ctx.other_info = (ao_to_atom, wrapper, shortname, to_transpose)
        return res

    @staticmethod
    def backward(ctx, grad_res: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        # grad_res: (*, nao, ngrid)
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
                dout_dcoeff = _EvalGTO.apply(*u_wrapper.params,
                                             rgrid, ao_to_atom, u_wrapper, shortname, False)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao = torch.gather(coeffs, dim=-1, index=ao2shl)  # (nu_ao)
                dout_dcoeff = dout_dcoeff / coeffs_ao[:, None]
                grad_dcoeff = torch.einsum("...ur,...ur->u", u_grad_res, dout_dcoeff)  # (nu_ao)

                grad_coeffs.scatter_add_(dim=-1, index=ao2shl, src=grad_dcoeff)

            if alphas.requires_grad:
                grad_alphas = torch.zeros_like(alphas)

                new_sname = _get_evalgto_derivname(shortname, "a")
                # (..., nu_ao, ngrid)
                dout_dalpha = _EvalGTO.apply(*u_wrapper.params, rgrid,
                                             ao_to_atom, u_wrapper, new_sname, False)

                alphas_ao = torch.gather(alphas, dim=-1, index=ao2shl)  # (nu_ao)
                grad_dalpha = -torch.einsum("...ur,...ur->u", u_grad_res, dout_dalpha)

                grad_alphas.scatter_add_(dim=-1, index=ao2shl, src=grad_dalpha)

        # calculate the gradient w.r.t. basis' pos and rgrid
        grad_pos = None
        grad_rgrid = None
        if rgrid.requires_grad or pos.requires_grad:
            opsname = _get_evalgto_derivname(shortname, "r")
            dresdr = _EvalGTO.apply(*ctx.saved_tensors,
                                    ao_to_atom, wrapper, opsname, False)  # (ndim, *, nao, ngrid)
            grad_r = dresdr * grad_res  # (ndim, *, nao, ngrid)

            if rgrid.requires_grad:
                grad_rgrid = grad_r.reshape(dresdr.shape[0], -1, dresdr.shape[-1])
                grad_rgrid = grad_rgrid.sum(dim=1).transpose(-2, -1)  # (ngrid, ndim)

            if pos.requires_grad:
                grad_rao = torch.movedim(grad_r, -2, 0)  # (nao, ndim, *, ngrid)
                grad_rao = -grad_rao.reshape(*grad_rao.shape[:2], -1).sum(dim=-1)  # (nao, ndim)
                grad_pos = torch.zeros_like(pos)  # (natom, ndim)
                grad_pos.scatter_add_(dim=0, index=ao_to_atom, src=grad_rao)

        return grad_coeffs, grad_alphas, grad_pos, grad_rgrid, \
            None, None, None, None, None, None

################### evaluator (direct interfact to libcgto) ###################
def gto_evaluator(wrapper: LibcintWrapper, shortname: str, rgrid: torch.Tensor,
                  to_transpose: bool):
    # NOTE: this function do not propagate gradient and should only be used
    # in this file only

    # rgrid: (ngrid, ndim)
    # returns: (*, nao, ngrid) if not to_transpose else (*, ngrid, nao)

    ngrid = rgrid.shape[0]
    nshells = len(wrapper)
    nao = wrapper.nao()
    opname = _get_evalgto_opname(shortname, wrapper.spherical)
    outshape = _get_evalgto_compshape(shortname) + (nao, ngrid)

    out = np.empty(outshape, dtype=np.float64)
    non0tab = np.ones(((ngrid + BLKSIZE - 1) // BLKSIZE, nshells),
                      dtype=np.int8)

    # TODO: check if we need to transpose it first?
    rgrid = rgrid.contiguous()
    coords = np.asarray(rgrid, dtype=np.float64, order='F')
    ao_loc = np.asarray(wrapper.full_shell_to_aoloc, dtype=np.int32)

    c_shls = (ctypes.c_int * 2)(*wrapper.shell_idxs)
    c_ngrid = ctypes.c_int(ngrid)

    # evaluate the orbital
    operator = getattr(CGTO(), opname)
    operator.restype = ctypes.c_double
    atm, bas, env = wrapper.atm_bas_env
    operator(c_ngrid, c_shls,
             np2ctypes(ao_loc),
             np2ctypes(out),
             np2ctypes(coords),
             np2ctypes(non0tab),
             np2ctypes(atm), int2ctypes(atm.shape[0]),
             np2ctypes(bas), int2ctypes(bas.shape[0]),
             np2ctypes(env))

    if to_transpose:
        out = np.ascontiguousarray(np.moveaxis(out, -1, -2))

    out_tensor = torch.as_tensor(out, dtype=wrapper.dtype, device=wrapper.device)
    return out_tensor

def _get_evalgto_opname(shortname: str, spherical: bool) -> str:
    # returns the complete name of the evalgto operation
    sname = ("_" + shortname) if (shortname != "") else ""
    suffix = "_sph" if spherical else "_cart"
    return "GTOval%s%s" % (sname, suffix)

def _get_evalgto_compshape(shortname: str) -> Tuple[int, ...]:
    # returns the component shape of the evalgto function

    # count "ip" only at the beginning
    n_ip = len(re.findall(r"^(?:ip)*(?:ip)?", shortname)[0]) // 2
    return (NDIM, ) * n_ip

def _get_evalgto_derivname(shortname: str, derivmode: str):
    if derivmode == "r":
        return "ip%s" % shortname
    elif derivmode == "a":
        return "%srr" % shortname
    else:
        raise RuntimeError("Unknown derivmode: %s" % derivmode)
