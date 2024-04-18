from typing import Optional, List, Tuple, Callable, Dict
import ctypes
import copy
import operator
from functools import reduce
import numpy as np
import torch
from dqc.hamilton.intor.lcintwrap import LibcintWrapper
from dqc.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CINT, CGTO
from dqc.hamilton.intor.namemgr import IntorNameManager

__all__ = ["int1e", "int3c2e", "int2e",
           "overlap", "kinetic", "nuclattr", "elrep", "coul2c", "coul3c"]

# integrals
def int1e(shortname: str, wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None, *,
          # additional options for some specific integrals
          rinv_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
    # 2-centre 1-electron integral

    # check and set the other parameters
    other1 = _check_and_set(wrapper, other)

    # set the rinv_pos arguments
    if "rinv" in shortname:
        assert isinstance(rinv_pos, torch.Tensor), "The keyword rinv_pos must be specified"
    else:
        # don't really care, it will be ignored
        rinv_pos = torch.zeros(1, dtype=wrapper.dtype, device=wrapper.device)

    return _Int2cFunction.apply(*wrapper.params,
                                rinv_pos,
                                [wrapper, other1],
                                IntorNameManager("int1e", shortname))

def int2c2e(shortname: str, wrapper: LibcintWrapper,
            other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    2-centre 2-electron integrals where the `wrapper` and `other1` correspond
    to the first electron, and `other2` corresponds to another electron.
    The returned indices are sorted based on `wrapper`, `other1`, and `other2`.
    The available shortname: "ar12"
    """

    # don't really care, it will be ignored
    rinv_pos = torch.zeros(1, dtype=wrapper.dtype, device=wrapper.device)

    # check and set the others
    otherw = _check_and_set(wrapper, other)
    return _Int2cFunction.apply(
        *wrapper.params,
        rinv_pos,
        [wrapper, otherw],
        IntorNameManager("int2c2e", shortname))

def int3c2e(shortname: str, wrapper: LibcintWrapper,
            other1: Optional[LibcintWrapper] = None,
            other2: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    3-centre 2-electron integrals where the `wrapper` and `other1` correspond
    to the first electron, and `other2` corresponds to another electron.
    The returned indices are sorted based on `wrapper`, `other1`, and `other2`.
    The available shortname: "ar12"
    """

    # check and set the others
    other1w = _check_and_set(wrapper, other1)
    other2w = _check_and_set(wrapper, other2)
    return _Int3cFunction.apply(
        *wrapper.params,
        [wrapper, other1w, other2w],
        IntorNameManager("int3c2e", shortname))

def int2e(shortname: str, wrapper: LibcintWrapper,
          other1: Optional[LibcintWrapper] = None,
          other2: Optional[LibcintWrapper] = None,
          other3: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    4-centre 2-electron integrals where the `wrapper` and `other1` correspond
    to the first electron, and `other2` and `other3` correspond to another
    electron.
    The returned indices are sorted based on `wrapper`, `other1`, `other2`, and `other3`.
    The available shortname: "ar12b"
    """

    # check and set the others
    other1w = _check_and_set(wrapper, other1)
    other2w = _check_and_set(wrapper, other2)
    other3w = _check_and_set(wrapper, other3)
    return _Int4cFunction.apply(
        *wrapper.params,
        [wrapper, other1w, other2w, other3w],
        IntorNameManager("int2e", shortname))

# shortcuts
def overlap(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    return int1e("ovlp", wrapper, other=other)

def kinetic(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    return int1e("kin", wrapper, other=other)

def nuclattr(wrapper: LibcintWrapper, other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    if not wrapper.fracz:
        return int1e("nuc", wrapper, other=other)
    else:
        res = torch.tensor([])
        allpos_params = wrapper.params[-1]
        for i in range(wrapper.natoms):
            y = int1e("rinv", wrapper, other=other, rinv_pos=allpos_params[i]) * \
                (-wrapper.atombases[i].atomz)
            res = y if (i == 0) else (res + y)
        return res

def elrep(wrapper: LibcintWrapper,
          other1: Optional[LibcintWrapper] = None,
          other2: Optional[LibcintWrapper] = None,
          other3: Optional[LibcintWrapper] = None,
          ) -> torch.Tensor:
    return int2e("ar12b", wrapper, other1, other2, other3)

def coul2c(wrapper: LibcintWrapper,
           other: Optional[LibcintWrapper] = None,
           ) -> torch.Tensor:
    return int2c2e("r12", wrapper, other)

def coul3c(wrapper: LibcintWrapper,
           other1: Optional[LibcintWrapper] = None,
           other2: Optional[LibcintWrapper] = None,
           ) -> torch.Tensor:
    return int3c2e("ar12", wrapper, other1, other2)

# misc functions
def _check_and_set(wrapper: LibcintWrapper, other: Optional[LibcintWrapper]) -> LibcintWrapper:
    # check the value and set the default value of "other" in the integrals
    if other is not None:
        atm0, bas0, env0 = wrapper.atm_bas_env
        atm1, bas1, env1 = other.atm_bas_env
        msg = ("Argument `other*` does not have the same parent as the wrapper. "
               "Please do `LibcintWrapper.concatenate` on those wrappers first.")
        assert id(atm0) == id(atm1), msg
        assert id(bas0) == id(bas1), msg
        assert id(env0) == id(env1), msg
    else:
        other = wrapper
    assert isinstance(other, LibcintWrapper)
    return other

############### pytorch functions ###############
class _Int2cFunction(torch.autograd.Function):
    # wrapper class to provide the gradient of the 2-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                rinv_pos: torch.Tensor,
                wrappers: List[LibcintWrapper], int_nmgr: IntorNameManager) -> torch.Tensor:
        # allcoeffs: (ngauss_tot,)
        # allalphas: (ngauss_tot,)
        # allposs: (natom, ndim)
        # rinv_pos: (ndim,) if contains "rinv"
        #           rinv_pos is only meaningful if shortname contains "rinv"
        # In "rinv", rinv_pos becomes the centre
        # Wrapper0 and wrapper1 must have the same _atm, _bas, and _env.
        # The check should be done before calling this function.
        # Those tensors are not used directly in the forward calculation, but
        #   required for backward propagation
        assert len(wrappers) == 2

        if int_nmgr.rawopname == "rinv":
            assert rinv_pos.ndim == 1 and rinv_pos.shape[0] == NDIM
            with wrappers[0].centre_on_r(rinv_pos):
                out_tensor = Intor(int_nmgr, wrappers).calc()
        else:
            out_tensor = Intor(int_nmgr, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs, rinv_pos)
        ctx.other_info = (wrappers, int_nmgr)
        return out_tensor  # (..., nao0, nao1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        # grad_out: (..., nao0, nao1)
        allcoeffs, allalphas, allposs, \
            rinv_pos = ctx.saved_tensors
        wrappers, int_nmgr = ctx.other_info

        # gradient for all atomic positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            # get the integrals required for the derivatives
            sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1)]
            # new axes added to the dimension
            new_axes_pos = [int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in (0, 1)]
            int_fcn = lambda wrappers, namemgr: _Int2cFunction.apply(
                *ctx.saved_tensors, wrappers, namemgr)
            # list of tensors with shape: (ndim, ..., nao0, nao1)
            dout_dposs = _get_integrals(sname_derivs, wrappers, int_fcn, new_axes_pos)

            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *dout_dposs[0].shape[-2:])
            grad_out2 = grad_out.reshape(shape[1:])
            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            grad_dpos_i = -torch.einsum("sij,dsij->di", grad_out2, dout_dposs[0].reshape(shape))
            grad_dpos_j = -torch.einsum("sij,dsij->dj", grad_out2, dout_dposs[1].reshape(shape))

            # grad_allpossT is only a view of grad_allposs, so the operation below
            # also changes grad_allposs
            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom0, src=grad_dpos_i)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom1, src=grad_dpos_j)

            grad_allposs_nuc = torch.zeros_like(grad_allposs)
            if "nuc" == int_nmgr.rawopname:
                # allposs: (natoms, ndim)
                natoms = allposs.shape[0]
                sname_rinv = int_nmgr.shortname.replace("nuc", "rinv")
                int_nmgr_rinv = IntorNameManager(int_nmgr.int_type, sname_rinv)
                sname_derivs = [int_nmgr_rinv.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1)]
                new_axes_pos = [int_nmgr_rinv.get_intgl_deriv_newaxispos("ip", ib) for ib in (0, 1)]

                for i in range(natoms):
                    atomz = wrappers[0].atombases[i].atomz

                    # get the integrals
                    int_fcn = lambda wrappers, namemgr: _Int2cFunction.apply(
                        allcoeffs, allalphas, allposs, allposs[i],
                        wrappers, namemgr)
                    dout_datposs = _get_integrals(sname_derivs, wrappers, int_fcn,
                                                  new_axes_pos)  # (ndim, ..., nao, nao)

                    grad_datpos = grad_out * (dout_datposs[0] + dout_datposs[1])
                    grad_datpos = grad_datpos.reshape(grad_datpos.shape[0], -1).sum(dim=-1)
                    grad_allposs_nuc[i] = (-atomz) * grad_datpos

                grad_allposs += grad_allposs_nuc

        # gradient for the rinv_pos in rinv integral
        grad_rinv_pos: Optional[torch.Tensor] = None
        if rinv_pos.requires_grad and "rinv" == int_nmgr.rawopname:
            # rinv_pos: (ndim)
            # get the integrals for the derivatives
            sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1)]
            # new axes added to the dimension
            new_axes_pos = [int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in (0, 1)]
            int_fcn = lambda wrappers, namemgr: _Int2cFunction.apply(
                *ctx.saved_tensors, wrappers, namemgr)
            dout_datposs = _get_integrals(sname_derivs, wrappers, int_fcn, new_axes_pos)

            grad_datpos = grad_out * (dout_datposs[0] + dout_datposs[1])
            grad_rinv_pos = grad_datpos.reshape(grad_datpos.shape[0], -1).sum(dim=-1)

        # gradient for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper and mapping
            # uao2aos: list of (nu_ao0,), (nu_ao1,)
            u_wrappers_tup, uao2aos_tup = zip(*[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # get the uncontracted (gathered) grad_out
            u_grad_out = _gather_at_dims(grad_out, mapidxs=uao2aos, dims=[-2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()
            ao2shl1 = u_wrappers[1].ao_to_shell()

            # calculate the gradient w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)  # (ngauss)

                # get the uncontracted version of the integral
                dout_dcoeff = _Int2cFunction.apply(
                    *u_params, rinv_pos, u_wrappers, int_nmgr)  # (..., nu_ao0, nu_ao1)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1, index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1, index=ao2shl1)  # (nu_ao1)
                # divide done here instead of after scatter to make the 2nd gradient
                # calculation correct.
                # division can also be done after scatter for more efficient 1st grad
                # calculation, but it gives the wrong result for 2nd grad
                dout_dcoeff_i = dout_dcoeff / coeffs_ao0[:, None]
                dout_dcoeff_j = dout_dcoeff / coeffs_ao1

                # (nu_ao)
                grad_dcoeff_i = torch.einsum("...ij,...ij->i", u_grad_out, dout_dcoeff_i)
                grad_dcoeff_j = torch.einsum("...ij,...ij->j", u_grad_out, dout_dcoeff_j)
                # grad_dcoeff = grad_dcoeff_i + grad_dcoeff_j

                # scatter the grad
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl0, src=grad_dcoeff_i)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl1, src=grad_dcoeff_j)

            # calculate the gradient w.r.t. alphas
            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                u_int_fcn = lambda u_wrappers, int_nmgr: _Int2cFunction.apply(
                    *u_params, rinv_pos, u_wrappers, int_nmgr)

                # get the uncontracted integrals
                sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("rr", ib) for ib in (0, 1)]
                new_axes_pos = [int_nmgr.get_intgl_deriv_newaxispos("rr", ib) for ib in (0, 1)]
                dout_dalphas = _get_integrals(sname_derivs, u_wrappers, u_int_fcn, new_axes_pos)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_dalpha_i = -torch.einsum("...ij,...ij->i", u_grad_out, dout_dalphas[0])
                grad_dalpha_j = -torch.einsum("...ij,...ij->j", u_grad_out, dout_dalphas[1])
                # grad_dalpha = (grad_dalpha_i + grad_dalpha_j)  # (nu_ao)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl0, src=grad_dalpha_i)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl1, src=grad_dalpha_j)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            grad_rinv_pos, \
            None, None, None

class _Int3cFunction(torch.autograd.Function):
    # wrapper class for the 3-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                wrappers: List[LibcintWrapper],
                int_nmgr: IntorNameManager) -> torch.Tensor:

        assert len(wrappers) == 3

        out_tensor = Intor(int_nmgr, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.other_info = (wrappers, int_nmgr)
        return out_tensor  # (..., nao0, nao1, nao2)

    @staticmethod
    def backward(ctx, grad_out) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        # grad_out: (..., nao0, nao1, nao2)
        allcoeffs, allalphas, allposs = ctx.saved_tensors
        wrappers, int_nmgr = ctx.other_info
        naos = grad_out.shape[-3:]

        # calculate the gradient w.r.t. positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1, 2)]
            new_axes_pos = [int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in (0, 1, 2)]
            int_fcn = lambda wrappers, int_nmgr: _Int3cFunction.apply(
                *ctx.saved_tensors, wrappers, int_nmgr)
            dout_dposs = _get_integrals(sname_derivs, wrappers, int_fcn, new_axes_pos)

            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *naos)
            grad_out2 = grad_out.reshape(*shape[1:])
            grad_pos_a1 = -torch.einsum("dzijk,zijk->di", dout_dposs[0].reshape(*shape), grad_out2)
            grad_pos_a2 = -torch.einsum("dzijk,zijk->dj", dout_dposs[1].reshape(*shape), grad_out2)
            grad_pos_b1 = -torch.einsum("dzijk,zijk->dk", dout_dposs[2].reshape(*shape), grad_out2)

            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            ao_to_atom2 = wrappers[2].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom0, src=grad_pos_a1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom1, src=grad_pos_a2)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom2, src=grad_pos_b1)

        # gradients for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper, and expanded grad_out
            # uao2ao: (nu_ao)
            u_wrappers_tup, uao2aos_tup = zip(*[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # u_grad_out: (..., nu_ao0, nu_ao1, nu_ao2)
            u_grad_out = _gather_at_dims(grad_out, mapidxs=uao2aos, dims=[-3, -2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()  # (nu_ao0,)
            ao2shl1 = u_wrappers[1].ao_to_shell()
            ao2shl2 = u_wrappers[2].ao_to_shell()

            # calculate the grad w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)

                # (..., nu_ao0, nu_ao1, nu_ao2)
                dout_dcoeff = _Int3cFunction.apply(*u_params, u_wrappers, int_nmgr)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1, index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1, index=ao2shl1)  # (nu_ao1)
                coeffs_ao2 = torch.gather(allcoeffs, dim=-1, index=ao2shl2)  # (nu_ao2)
                # dout_dcoeff_*: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff_a1 = dout_dcoeff / coeffs_ao0[:, None, None]
                dout_dcoeff_a2 = dout_dcoeff / coeffs_ao1[:, None]
                dout_dcoeff_b1 = dout_dcoeff / coeffs_ao2

                # reduce the uncontracted integrations
                # grad_coeff_*: (nu_ao*)
                grad_coeff_a1 = torch.einsum("...ijk,...ijk->i", dout_dcoeff_a1, u_grad_out)
                grad_coeff_a2 = torch.einsum("...ijk,...ijk->j", dout_dcoeff_a2, u_grad_out)
                grad_coeff_b1 = torch.einsum("...ijk,...ijk->k", dout_dcoeff_b1, u_grad_out)
                # grad_coeff_all = grad_coeff_a1 + grad_coeff_a2 + grad_coeff_b1 + grad_coeff_b2

                # scatter to the coefficients
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl0, src=grad_coeff_a1)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl1, src=grad_coeff_a2)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl2, src=grad_coeff_b1)

            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                # get the uncontracted integrals
                sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("rr", ib) for ib in (0, 1, 2)]
                new_axes_pos = [int_nmgr.get_intgl_deriv_newaxispos("rr", ib) for ib in (0, 1, 2)]
                u_int_fcn = lambda u_wrappers, int_nmgr: _Int3cFunction.apply(
                    *u_params, u_wrappers, int_nmgr)
                dout_dalphas = _get_integrals(sname_derivs, u_wrappers, u_int_fcn, new_axes_pos)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_alpha_a1 = -torch.einsum("...ijk,...ijk->i", dout_dalphas[0], u_grad_out)
                grad_alpha_a2 = -torch.einsum("...ijk,...ijk->j", dout_dalphas[1], u_grad_out)
                grad_alpha_b1 = -torch.einsum("...ijk,...ijk->k", dout_dalphas[2], u_grad_out)
                # grad_alpha_all = (grad_alpha_a1 + grad_alpha_a2 + grad_alpha_b1 + grad_alpha_b2)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl0, src=grad_alpha_a1)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl1, src=grad_alpha_a2)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl2, src=grad_alpha_b1)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            None, None, None

class _Int4cFunction(torch.autograd.Function):
    # wrapper class for the 4-centre integrals
    @staticmethod
    def forward(ctx,  # type: ignore
                allcoeffs: torch.Tensor, allalphas: torch.Tensor, allposs: torch.Tensor,
                wrappers: List[LibcintWrapper],
                int_nmgr: IntorNameManager) -> torch.Tensor:

        assert len(wrappers) == 4

        out_tensor = Intor(int_nmgr, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.other_info = (wrappers, int_nmgr)
        return out_tensor  # (..., nao0, nao1, nao2, nao3)

    @staticmethod
    def backward(ctx, grad_out) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        # grad_out: (..., nao0, nao1, nao2, nao3)
        allcoeffs, allalphas, allposs = ctx.saved_tensors
        wrappers, int_nmgr = ctx.other_info
        naos = grad_out.shape[-4:]

        # calculate the gradient w.r.t. positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in range(4)]
            new_axes_pos = [int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in range(4)]
            int_fcn = lambda wrappers, int_nmgr: _Int4cFunction.apply(
                *ctx.saved_tensors, wrappers, int_nmgr)
            dout_dposs = _get_integrals(sname_derivs, wrappers, int_fcn, new_axes_pos)

            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *naos)
            grad_out2 = grad_out.reshape(*shape[1:])
            grad_pos_a1 = -torch.einsum("dzijkl,zijkl->di", dout_dposs[0].reshape(*shape), grad_out2)
            grad_pos_a2 = -torch.einsum("dzijkl,zijkl->dj", dout_dposs[1].reshape(*shape), grad_out2)
            grad_pos_b1 = -torch.einsum("dzijkl,zijkl->dk", dout_dposs[2].reshape(*shape), grad_out2)
            grad_pos_b2 = -torch.einsum("dzijkl,zijkl->dl", dout_dposs[3].reshape(*shape), grad_out2)

            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            ao_to_atom2 = wrappers[2].ao_to_atom().expand(ndim, -1)
            ao_to_atom3 = wrappers[3].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom0, src=grad_pos_a1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom1, src=grad_pos_a2)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom2, src=grad_pos_b1)
            grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom3, src=grad_pos_b2)

        # gradients for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper, and expanded grad_out
            # uao2ao: (nu_ao)
            u_wrappers_tup, uao2aos_tup = zip(*[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # u_grad_out: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
            u_grad_out = _gather_at_dims(grad_out, mapidxs=uao2aos, dims=[-4, -3, -2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()  # (nu_ao0,)
            ao2shl1 = u_wrappers[1].ao_to_shell()
            ao2shl2 = u_wrappers[2].ao_to_shell()
            ao2shl3 = u_wrappers[3].ao_to_shell()

            # calculate the grad w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)

                # (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff = _Int4cFunction.apply(*u_params, u_wrappers, int_nmgr)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1, index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1, index=ao2shl1)  # (nu_ao1)
                coeffs_ao2 = torch.gather(allcoeffs, dim=-1, index=ao2shl2)  # (nu_ao2)
                coeffs_ao3 = torch.gather(allcoeffs, dim=-1, index=ao2shl3)  # (nu_ao3)
                # dout_dcoeff_*: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff_a1 = dout_dcoeff / coeffs_ao0[:, None, None, None]
                dout_dcoeff_a2 = dout_dcoeff / coeffs_ao1[:, None, None]
                dout_dcoeff_b1 = dout_dcoeff / coeffs_ao2[:, None]
                dout_dcoeff_b2 = dout_dcoeff / coeffs_ao3

                # reduce the uncontracted integrations
                # grad_coeff_*: (nu_ao*)
                grad_coeff_a1 = torch.einsum("...ijkl,...ijkl->i", dout_dcoeff_a1, u_grad_out)
                grad_coeff_a2 = torch.einsum("...ijkl,...ijkl->j", dout_dcoeff_a2, u_grad_out)
                grad_coeff_b1 = torch.einsum("...ijkl,...ijkl->k", dout_dcoeff_b1, u_grad_out)
                grad_coeff_b2 = torch.einsum("...ijkl,...ijkl->l", dout_dcoeff_b2, u_grad_out)
                # grad_coeff_all = grad_coeff_a1 + grad_coeff_a2 + grad_coeff_b1 + grad_coeff_b2

                # scatter to the coefficients
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl0, src=grad_coeff_a1)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl1, src=grad_coeff_a2)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl2, src=grad_coeff_b1)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl3, src=grad_coeff_b2)

            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                # get the uncontracted integrals
                sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("rr", ib) for ib in range(4)]
                new_axes_pos = [int_nmgr.get_intgl_deriv_newaxispos("rr", ib) for ib in range(4)]
                u_int_fcn = lambda u_wrappers, int_nmgr: _Int4cFunction.apply(
                    *u_params, u_wrappers, int_nmgr)
                dout_dalphas = _get_integrals(sname_derivs, u_wrappers, u_int_fcn, new_axes_pos)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_alpha_a1 = -torch.einsum("...ijkl,...ijkl->i", dout_dalphas[0], u_grad_out)
                grad_alpha_a2 = -torch.einsum("...ijkl,...ijkl->j", dout_dalphas[1], u_grad_out)
                grad_alpha_b1 = -torch.einsum("...ijkl,...ijkl->k", dout_dalphas[2], u_grad_out)
                grad_alpha_b2 = -torch.einsum("...ijkl,...ijkl->l", dout_dalphas[3], u_grad_out)
                # grad_alpha_all = (grad_alpha_a1 + grad_alpha_a2 + grad_alpha_b1 + grad_alpha_b2)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl0, src=grad_alpha_a1)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl1, src=grad_alpha_a2)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl2, src=grad_alpha_b1)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl3, src=grad_alpha_b2)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            None, None, None

################### integrator (direct interface to libcint) ###################

# Optimizer class
class _cintoptHandler(ctypes.c_void_p):
    def __del__(self):
        try:
            CGTO().CINTdel_optimizer(ctypes.byref(self))
        except AttributeError:
            pass

class Intor(object):
    def __init__(self, int_nmgr: IntorNameManager, wrappers: List[LibcintWrapper]):
        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        self.int_type = int_nmgr.int_type
        self.atm, self.bas, self.env = wrapper0.atm_bas_env
        self.wrapper0 = wrapper0
        self.int_nmgr = int_nmgr
        self.wrapper_uniqueness = _get_uniqueness([id(w) for w in wrappers])

        # get the operator
        opname = int_nmgr.get_intgl_name(wrapper0.spherical)
        self.op = getattr(CINT(), opname)
        self.optimizer = _get_intgl_optimizer(opname, self.atm, self.bas, self.env)

        # prepare the output
        comp_shape = int_nmgr.get_intgl_components_shape()
        self.outshape = comp_shape + tuple(w.nao() for w in wrappers)
        self.ncomp = reduce(operator.mul, comp_shape, 1)
        self.shls_slice = sum((w.shell_idxs for w in wrappers), ())
        self.integral_done = False

    def calc(self) -> torch.Tensor:
        assert not self.integral_done
        self.integral_done = True
        if self.int_type == "int1e" or self.int_type == "int2c2e":
            return self._int2c()
        elif self.int_type == "int3c2e":
            return self._int3c()
        elif self.int_type == "int2e":
            return self._int4c()
        else:
            raise ValueError("Unknown integral type: %s" % self.int_type)

    def _int2c(self) -> torch.Tensor:
        # performing 2-centre integrals with libcint
        drv = CGTO().GTOint2c
        outshape = self.outshape
        out = np.empty((*outshape[:-2], outshape[-1], outshape[-2]), dtype=np.float64)
        drv(self.op,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.ncomp),
            ctypes.c_int(0),  # do not assume hermitian
            (ctypes.c_int * len(self.shls_slice))(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm), int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas), int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out = np.swapaxes(out, -2, -1)
        # TODO: check if we need to do the lines below for 3rd order grad and higher
        # if out.ndim > 2:
        #     out = np.moveaxis(out, -3, 0)
        return self._to_tensor(out)

    def _int3c(self) -> torch.Tensor:
        # performing 3-centre integrals with libcint
        drv = CGTO().GTOnr3c_drv
        fill = CGTO().GTOnr3c_fill_s1
        # TODO: create optimizer without the 3rd index like in
        # https://github.com/pyscf/pyscf/blob/e833b9a4fd5fb24a061721e5807e92c44bb66d06/pyscf/gto/moleintor.py#L538
        outsh = self.outshape
        out = np.empty((*outsh[:-3], outsh[-1], outsh[-2], outsh[-3]), dtype=np.float64)
        drv(self.op, fill,
            out.ctypes.data_as(ctypes.c_void_p),
            int2ctypes(self.ncomp),
            (ctypes.c_int * len(self.shls_slice))(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm), int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas), int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out = np.swapaxes(out, -3, -1)
        return self._to_tensor(out)

    def _int4c(self) -> torch.Tensor:
        # performing 4-centre integrals with libcint
        symm = self.int_nmgr.get_intgl_symmetry(self.wrapper_uniqueness)
        outshape = symm.get_reduced_shape(self.outshape)

        out = np.empty(outshape, dtype=np.float64)

        drv = CGTO().GTOnr2e_fill_drv
        fill = getattr(CGTO(), "GTOnr2e_fill_%s" % symm.code)
        prescreen = ctypes.POINTER(ctypes.c_void_p)()
        drv(self.op, fill, prescreen,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.ncomp),
            (ctypes.c_int * 8)(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm), int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas), int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out = symm.reconstruct_array(out, self.outshape)
        return self._to_tensor(out)

    def _to_tensor(self, out: np.ndarray) -> torch.Tensor:
        # convert the numpy array to the appropriate tensor
        return torch.as_tensor(out, dtype=self.wrapper0.dtype,
                               device=self.wrapper0.device)

def _get_intgl_optimizer(opname: str,
                         atm: np.ndarray, bas: np.ndarray, env: np.ndarray)\
                         -> ctypes.c_void_p:
    # get the optimizer of the integrals
    # setup the optimizer
    cintopt = ctypes.POINTER(ctypes.c_void_p)()
    optname = opname.replace("_cart", "").replace("_sph", "") + "_optimizer"
    copt = getattr(CINT(), optname)
    copt(ctypes.byref(cintopt),
         np2ctypes(atm), int2ctypes(atm.shape[0]),
         np2ctypes(bas), int2ctypes(bas.shape[0]),
         np2ctypes(env))
    opt = ctypes.cast(cintopt, _cintoptHandler)
    return opt

############### name derivation manager functions ###############
def _get_integrals(int_nmgrs: List[IntorNameManager],
                   wrappers: List[LibcintWrapper],
                   int_fcn: Callable[[List[LibcintWrapper], IntorNameManager], torch.Tensor],
                   new_axes_pos: List[int]) \
                   -> List[torch.Tensor]:
    # Return the list of tensors of the integrals given by the list of integral names.
    # Int_fcn is the integral function that receives the name and returns the results.
    # If new_axes_pos is specified, then move the new axes to 0, otherwise, just leave
    # it as it is

    res: List[torch.Tensor] = []
    # indicating if the integral is available in the libcint-generated file
    int_avail: List[bool] = [False] * len(int_nmgrs)

    for i in range(len(int_nmgrs)):
        res_i: Optional[torch.Tensor] = None

        # check if the integral can be calculated from the previous results
        for j in range(i - 1, -1, -1):

            # check the integral names equivalence
            transpose_path = int_nmgrs[j].get_transpose_path_to(int_nmgrs[i])
            if transpose_path is not None:

                # if the swapped wrappers remain unchanged, then just use the
                # transposed version of the previous version
                # TODO: think more about this (do we need to use different
                # transpose path? e.g. transpose_path[::-1])
                twrappers = _swap_list(wrappers, transpose_path)
                if twrappers == wrappers:
                    res_i = _transpose(res[j], transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # otherwise, use the swapped integral with the swapped wrappers,
                # only if the integral is available in the libcint-generated
                # files
                elif int_avail[j]:
                    res_i = int_fcn(twrappers, int_nmgrs[j])
                    res_i = _transpose(res_i, transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # if the integral is not available, then continue the searching
                else:
                    continue

        if res_i is None:
            try:
                # successfully executing the line below indicates that the integral
                # is available in the libcint-generated files
                res_i = int_fcn(wrappers, int_nmgrs[i])
            except AttributeError:
                msg = "The integral %s is not available from libcint, please add it" % int_nmgrs[i].fullname
                raise AttributeError(msg)

            int_avail[i] = True

        res.append(res_i)

    # move the new axes (if any) to dimension 0
    assert res_i is not None
    for i in range(len(res)):
        if new_axes_pos[i] is not None:
            res[i] = torch.movedim(res[i], new_axes_pos[i], 0)

    return res

def _transpose(a: torch.Tensor, axes: List[Tuple[int, int]]) -> torch.Tensor:
    # perform the transpose of two axes for tensor a
    for axis2 in axes:
        a = a.transpose(*axis2)
    return a

def _swap_list(a: List, swaps: List[Tuple[int, int]]) -> List:
    # swap the elements according to the swaps input
    res = copy.copy(a)  # shallow copy
    for idxs in swaps:
        res[idxs[0]], res[idxs[1]] = res[idxs[1]], res[idxs[0]]  # swap the elements
    return res

def _gather_at_dims(inp: torch.Tensor, mapidxs: List[torch.Tensor],
                    dims: List[int]) -> torch.Tensor:
    # expand inp in the dimension dim by gathering values based on the given
    # mapping indices

    # mapidx: (nnew,) with value from 0 to nold - 1
    # inp: (..., nold, ...)
    # out: (..., nnew, ...)
    out = inp
    for (dim, mapidx) in zip(dims, mapidxs):
        if dim < 0:
            dim = out.ndim + dim
        map2 = mapidx[(...,) + (None,) * (out.ndim - 1 - dim)]
        map2 = map2.expand(*out.shape[:dim], -1, *out.shape[dim + 1:])
        out = torch.gather(out, dim=dim, index=map2)
    return out

def _get_uniqueness(a: List) -> List[int]:
    # get the uniqueness pattern from the list, e.g. _get_uniqueness([1, 1, 2, 3, 2])
    # will return [0, 0, 1, 2, 1]
    s: Dict = {}
    res: List[int] = []
    i = 0
    for elmt in a:
        if elmt in s:
            res.append(s[elmt])
        else:
            s[elmt] = i
            res.append(i)
            i += 1
    return res