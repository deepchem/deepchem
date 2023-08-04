"""
Density Functional Theory Data
Derived from: https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/entry.py
"""
from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import List, Dict, Optional
import numpy as np
import torch
# dqc depend
import dqc
from dqc.system.mol import Mol
from dqc.system.base_system import BaseSystem
from dqc.grid.base_grid import BaseGrid
from deepchem.utils.dftutils import KSCalc

from typing import List, Optional, Union, overload, Tuple, Type
import warnings
import torch
import xitorch as xt
import dqc.hamilton.intor as intor
from dqc.df.base_df import BaseDF
from dqc.df.dfmol import DFMol
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.orbparams import BaseOrbParams, QROrbParams, MatExpOrbParams
from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam, DensityFitInfo
from dqc.grid.base_grid import BaseGrid
from dqc.xc.base_xc import BaseXC
from dqc.utils.cache import Cache
from dqc.utils.mem import chunkify, get_dtype_memsize
from dqc.utils.config import config
from dqc.utils.misc import logger

from typing import List, Union, Optional, Tuple, Dict
import warnings
from dqc.hamilton.base_hamilton import BaseHamilton
from dqc.hamilton.hcgto import HamiltonCGTO
from dqc.system.base_system import BaseSystem
from dqc.grid.base_grid import BaseGrid
from dqc.grid.factory import get_predefined_grid
from dqc.utils.datastruct import CGTOBasis, AtomCGTOBasis, SpinParam, ZType, \
                                 is_z_float, BasisInpType, DensityFitInfo, \
                                 AtomZsType, AtomPosType
from dqc.utils.periodictable import get_atomz, get_atom_mass
from dqc.utils.safeops import occnumber, safe_cdist
from dqc.api.loadbasis import loadbasis
from dqc.api.parser import parse_moldesc
from dqc.utils.cache import Cache
from dqc.utils.misc import logger

from contextlib import contextmanager
from typing import List, Tuple, Iterator, Optional, Dict
import copy
from dqc.utils.datastruct import AtomCGTOBasis, CGTOBasis
from dqc.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CINT
from dqc.hamilton.intor.lattice import Lattice
from dqc.utils.misc import memoize_method
from dqc.hamilton.intor.lcintwrap import LibcintWrapper
from dqc.hamilton.intor.gtoeval import _EvalGTO
from dqc.hamilton.intor.gtoeval import _get_evalgto_opname, _get_evalgto_compshape
import ctypes
from dqc.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CGTO

BLKSIZE = 128


class OrbitalOrthogonalizer(dqc.hamilton.orbconverter.OrbitalOrthogonalizer):

    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 2 dimensions of the matrix with shape (..., nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2)
        """
        self._orthozer = self._orthozer.to("cuda:0")
        res = self._orthozer.transpose(
            -2, -1).conj() @ mat.to("cuda:0") @ self._orthozer
        return res


class IdentityOrbConverter(dqc.hamilton.orbconverter.IdentityOrbConverter):

    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 2 dimensions of the matrix with shape (..., nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2)
        """
        return mat.to("cuda:0")


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
    coords = np.asarray(rgrid.clone().detach().cpu(),
                        dtype=np.float64,
                        order='F')
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


class EvalGTO(_EvalGTO):

    def forward(
            ctx,  # type: ignore
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

        res = gto_evaluator(wrapper, shortname, rgrid,
                            to_transpose)  # (*, nao, ngrid)
        ctx.save_for_backward(coeffs, alphas, pos, rgrid)
        ctx.other_info = (ao_to_atom, wrapper, shortname, to_transpose)
        return res


def evl(shortname: str,
        wrapper: LibcintWrapper,
        rgrid: torch.Tensor,
        *,
        to_transpose: bool = False) -> torch.Tensor:
    # expand ao_to_atom to have shape of (nao, ndim)
    ao_to_atom = wrapper.ao_to_atom().unsqueeze(-1).expand(-1, NDIM)

    # rgrid: (ngrid, ndim)
    return EvalGTO.apply(
        # tensors
        *wrapper.params,
        rgrid,

        # nontensors or int tensors
        ao_to_atom,
        wrapper,
        shortname,
        to_transpose)


def eval_gto(wrapper: LibcintWrapper,
             rgrid: torch.Tensor,
             *,
             to_transpose: bool = False) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (nao, ngrid)
    return evl("", wrapper, rgrid, to_transpose=to_transpose)


def eval_gradgto(wrapper: LibcintWrapper,
                 rgrid: torch.Tensor,
                 *,
                 to_transpose: bool = False) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (ndim, nao, ngrid)
    return evl("ip", wrapper, rgrid, to_transpose=to_transpose)


def eval_laplgto(wrapper: LibcintWrapper,
                 rgrid: torch.Tensor,
                 *,
                 to_transpose: bool = False) -> torch.Tensor:
    # rgrid: (ngrid, ndim)
    # return: (nao, ngrid)
    return evl("lapl", wrapper, rgrid, to_transpose=to_transpose)


class libcint(dqc.hamilton.intor.lcintwrap.LibcintWrapper):

    def __init__(self,
                 atombases: List[AtomCGTOBasis],
                 spherical: bool = True,
                 lattice: Optional[Lattice] = None) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._fracz = False
        self._natoms = len(atombases)
        self._lattice = lattice

        # get dtype and device for torch's tensors
        self.dtype = atombases[0].bases[0].alphas.dtype
        self.device = atombases[0].bases[0].alphas.device

        # construct _atm, _bas, and _env as well as the parameters
        ptr_env = 20  # initial padding from libcint
        atm_list: List[List[int]] = []
        env_list: List[float] = [0.0] * ptr_env
        bas_list: List[List[int]] = []
        allpos: List[torch.Tensor] = []
        allalphas: List[torch.Tensor] = []
        allcoeffs: List[torch.Tensor] = []
        allangmoms: List[int] = []
        shell_to_atom: List[int] = []
        ngauss_at_shell: List[int] = []
        gauss_to_shell: List[int] = []

        # constructing the triplet lists and also collecting the parameters
        nshells = 0
        ishell = 0
        for iatom, atombasis in enumerate(atombases):
            # construct the atom environment
            assert atombasis.pos.numel(
            ) == NDIM, "Please report this bug in Github"
            atomz = atombasis.atomz
            #                charge    ptr_coord, nucl model (unused for standard nucl model)
            atm_list.append([int(atomz), ptr_env, 1, ptr_env + NDIM, 0, 0])
            env_list.extend(atombasis.pos.detach())
            env_list.append(0.0)
            ptr_env += NDIM + 1

            # check if the atomz is fractional
            if isinstance(atomz, float) or \
                    (isinstance(atomz, torch.Tensor) and atomz.is_floating_point()):
                self._fracz = True

            # add the atom position into the parameter list
            # TODO: consider moving allpos into shell
            allpos.append(atombasis.pos.unsqueeze(0))

            nshells += len(atombasis.bases)
            shell_to_atom.extend([iatom] * len(atombasis.bases))

            # then construct the basis
            for shell in atombasis.bases:
                assert shell.alphas.shape == shell.coeffs.shape and shell.alphas.ndim == 1,\
                    "Please report this bug in Github"
                shell.wfnormalize_()
                ngauss = len(shell.alphas)
                #                iatom, angmom,       ngauss, ncontr, kappa, ptr_exp
                bas_list.append([
                    iatom,
                    shell.angmom,
                    ngauss,
                    1,
                    0,
                    ptr_env,
                    # ptr_coeffs,           unused
                    ptr_env + ngauss,
                    0
                ])
                env_list.extend(shell.alphas.detach())
                env_list.extend(shell.coeffs.detach())
                ptr_env += 2 * ngauss

                # add the alphas and coeffs to the parameters list
                allalphas.append(shell.alphas)
                allcoeffs.append(shell.coeffs)
                allangmoms.extend([shell.angmom] * ngauss)
                ngauss_at_shell.append(ngauss)
                gauss_to_shell.extend([ishell] * ngauss)
                ishell += 1

        # compile the parameters of this object
        self._allpos_params = torch.cat(allpos, dim=0)  # (natom, NDIM)
        self._allalphas_params = torch.cat(allalphas, dim=0)  # (ntot_gauss)
        self._allcoeffs_params = torch.cat(allcoeffs, dim=0)  # (ntot_gauss)
        self._allangmoms = torch.tensor(allangmoms,
                                        dtype=torch.int32,
                                        device=self.device)  # (ntot_gauss)
        self._gauss_to_shell = torch.tensor(gauss_to_shell,
                                            dtype=torch.int32,
                                            device=self.device)

        # convert the lists to numpy to make it contiguous (Python lists are not contiguous)
        self._atm = np.array(atm_list, dtype=np.int32, order="C")
        self._bas = np.array(bas_list, dtype=np.int32, order="C")
        print(env_list)
        self._env = np.array([((element.clone()).detach()).cpu()
                              if type(element) == torch.Tensor else element
                              for element in env_list],
                             dtype=np.float64,
                             order='C')
        print(self._env)
        # construct the full shell mapping
        shell_to_aoloc = [0]
        ao_to_shell: List[int] = []
        ao_to_atom: List[int] = []
        for i in range(nshells):
            nao_at_shell_i = self._nao_at_shell(i)
            shell_to_aoloc_i = shell_to_aoloc[-1] + nao_at_shell_i
            shell_to_aoloc.append(shell_to_aoloc_i)
            ao_to_shell.extend([i] * nao_at_shell_i)
            ao_to_atom.extend([shell_to_atom[i]] * nao_at_shell_i)

        self._ngauss_at_shell_list = ngauss_at_shell
        self._shell_to_aoloc = np.array(shell_to_aoloc, dtype=np.int32)
        self._shell_idxs = (0, nshells)
        self._ao_to_shell = torch.tensor(ao_to_shell,
                                         dtype=torch.long,
                                         device=self.device)
        self._ao_to_atom = torch.tensor(ao_to_atom,
                                        dtype=torch.long,
                                        device=self.device)


class Hamiltonz(dqc.hamilton.hcgto.HamiltonCGTO):

    def __init__(self,
                 atombases: List[AtomCGTOBasis],
                 spherical: bool = True,
                 df: Optional[DensityFitInfo] = None,
                 efield: Optional[Tuple[torch.Tensor, ...]] = None,
                 vext: Optional[torch.Tensor] = None,
                 cache: Optional[Cache] = None,
                 orthozer: bool = True,
                 aoparamzer: str = "qr") -> None:
        self.atombases = atombases
        self.spherical = spherical
        self.libcint_wrapper = libcint(atombases, spherical)
        self.dtype = self.libcint_wrapper.dtype
        self.device = self.libcint_wrapper.device

        # set up the orbital converter
        ovlp = intor.overlap(self.libcint_wrapper)
        if orthozer:
            self._orthozer = OrbitalOrthogonalizer(ovlp)
        else:
            self._orthozer = IdentityOrbConverter(ovlp)

        # set up the orbital parameterized
        if aoparamzer == "qr":
            self._orbparam: Type[BaseOrbParams] = QROrbParams
        elif aoparamzer == "matexp":
            warnings.warn(
                "Parametrization with matrix exponential is still at the experimental stage."
            )
            self._orbparam = MatExpOrbParams
        else:
            aoparam_opts = ["qr", "matexp"]
            raise RuntimeError(
                f"Unknown ao parameterizer: {aoparamzer}. Available options are: {aoparam_opts}"
            )

        # set up the density matrix
        self._dfoptions = df
        if df is None:
            self._df: Optional[DFMol] = None
        else:
            self._df = DFMol(df,
                             wrapper=self.libcint_wrapper,
                             orthozer=self._orthozer)

        self._efield = efield
        self._vext = vext
        self.is_grid_set = False
        self.is_ao_set = False
        self.is_grad_ao_set = False
        self.is_lapl_ao_set = False
        self.xc: Optional[BaseXC] = None
        self.xcfamily = 1
        self.is_built = False

        # initialize cache
        self._cache = cache if cache is not None else Cache.get_dummy()
        self._cache.add_cacheable_params(
            ["overlap", "kinetic", "nuclattr", "efield0"])
        if self._df is None:
            self._cache.add_cacheable_params(["elrep"])

    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        # save the family and save the xc
        self.xc = xc
        if xc is None:
            self.xcfamily = 1
        else:
            self.xcfamily = xc.family

        # save the grid
        self.grid = grid
        self.rgrid = grid.get_rgrid()
        assert grid.coord_type == "cart"

        # setup the basis as a spatial function
        logger.log("Calculating the basis values in the grid")
        self.is_ao_set = True
        self.basis = eval_gto(self.libcint_wrapper,
                              self.rgrid,
                              to_transpose=True).to("cuda:0")  # (ngrid, nao)
        self.dvolume = self.grid.get_dvolume().to("cuda:0")
        self.basis_dvolume = self.basis * self.dvolume.unsqueeze(
            -1)  # (ngrid, nao)

        if self.xcfamily == 1:  # LDA
            return

        # setup the gradient of the basis
        logger.log("Calculating the basis gradient values in the grid")
        self.is_grad_ao_set = True
        # (ndim, nao, ngrid)
        self.grad_basis = intor.eval_gradgto(self.libcint_wrapper,
                                             self.rgrid,
                                             to_transpose=True)
        if self.xcfamily == 2:  # GGA
            return

        # setup the laplacian of the basis
        self.is_lapl_ao_set = True
        logger.log("Calculating the basis laplacian values in the grid")
        self.lapl_basis = intor.eval_laplgto(self.libcint_wrapper,
                                             self.rgrid,
                                             to_transpose=True)  # (nao, ngrid)

    def get_vxc(self, dm):
        # dm: (*BD, nao, nao)
        assert self.xc is not None, "Please call .setup_grid with the xc object"
        densinfo = SpinParam.apply_fcn(lambda dm_: self._dm2densinfo(dm_),
                                       dm)  # value: (*BD, nr)
        potinfo = self.xc.get_vxc(densinfo)  # value: (*BD, nr)
        vxc_linop = SpinParam.apply_fcn(
            lambda potinfo_: self._get_vxc_from_potinfo(potinfo_), potinfo)
        return vxc_linop

    def get_elrep(self, dm: torch.Tensor) -> xt.LinearOperator:
        # dm: (*BD, nao, nao)
        # elrep_mat: (nao, nao, nao, nao)
        # return: (*BD, nao, nao)
        if self._df is None:
            self.el_mat = self.el_mat.to("cuda:0")
            mat = torch.einsum("...ij,ijkl->...kl", dm, self.el_mat)
            mat = (mat +
                   mat.transpose(-2, -1)) * 0.5  # reduce numerical instability
            return xt.LinearOperator.m(mat, is_hermitian=True)
        else:
            elrep = self._df.get_elrep(dm)
            return elrep

    def _dm2densinfo(self, dm: torch.Tensor) -> ValGrad:
        # dm: (*BD, nao, nao), Hermitian
        # family: 1 for LDA, 2 for GGA, 4 for MGGA
        # self.basis: (ngrid, nao)
        # self.grad_basis: (ndim, ngrid, nao)

        ngrid = self.basis.shape[-2]
        batchshape = dm.shape[:-2]
        dm = dm.to("cuda:0")

        # dm @ ao will be used in every case
        dmdmt = (dm + dm.transpose(-2, -1)) * 0.5  # (*BD, nao2, nao2)
        # convert it back to dm in the cgto basis
        dmdmt = self._orthozer.unconvert_dm(dmdmt)

        # prepare the densinfo components
        dens = torch.empty((*batchshape, ngrid),
                           dtype=self.dtype,
                           device=self.device)
        gdens: Optional[torch.Tensor] = None
        lapldens: Optional[torch.Tensor] = None
        kindens: Optional[torch.Tensor] = None
        if self.xcfamily == 2 or self.xcfamily == 4:  # GGA or MGGA
            gdens = torch.empty((*dm.shape[:-2], 3, ngrid),
                                dtype=self.dtype,
                                device=self.device)  # (..., ndim, ngrid)
        if self.xcfamily == 4:  # MGGA
            lapldens = torch.empty((*batchshape, ngrid),
                                   dtype=self.dtype,
                                   device=self.device)
            kindens = torch.empty((*batchshape, ngrid),
                                  dtype=self.dtype,
                                  device=self.device)

        # It is faster to split into chunks than evaluating a single big chunk
        maxnumel = config.CHUNK_MEMORY // get_dtype_memsize(self.basis)
        for basis, ioff, iend in chunkify(self.basis, dim=0, maxnumel=maxnumel):
            # basis: (ngrid2, nao)
            dmao = torch.matmul(basis, dmdmt)  # (ngrid2, nao)
            dens[..., ioff:iend] = torch.einsum("...ri,ri->...r", dmao, basis)

            if self.xcfamily == 2 or self.xcfamily == 4:  # GGA or MGGA
                assert gdens is not None
                if not self.is_grad_ao_set:
                    msg = "Please call `setup_grid(grid, gradlevel>=1)` to calculate the density gradient"
                    raise RuntimeError(msg)

                # summing it 3 times is faster than applying the d-axis directly
                grad_basis0 = self.grad_basis[0, ioff:iend, :]  # (ngrid2, nao)
                grad_basis1 = self.grad_basis[1, ioff:iend, :]
                grad_basis2 = self.grad_basis[2, ioff:iend, :]

                gdens[..., 0, ioff:iend] = torch.einsum("...ri,ri->...r", dmao,
                                                        grad_basis0) * 2
                gdens[..., 1, ioff:iend] = torch.einsum("...ri,ri->...r", dmao,
                                                        grad_basis1) * 2
                gdens[..., 2, ioff:iend] = torch.einsum("...ri,ri->...r", dmao,
                                                        grad_basis2) * 2

            if self.xcfamily == 4:
                assert lapldens is not None
                assert kindens is not None
                # calculate the laplacian of the density and kinetic energy density at the grid
                if not self.is_lapl_ao_set:
                    msg = "Please call `setup_grid(grid, gradlevel>=2)` to calculate the density gradient"
                    raise RuntimeError(msg)

                lapl_basis_cat = self.lapl_basis[ioff:iend, :]
                lapl_basis = torch.einsum("...ri,ri->...r", dmao,
                                          lapl_basis_cat)
                grad_grad = torch.einsum("...ri,ri->...r",
                                         torch.matmul(grad_basis0, dmdmt),
                                         grad_basis0)
                grad_grad += torch.einsum("...ri,ri->...r",
                                          torch.matmul(grad_basis1, dmdmt),
                                          grad_basis1)
                grad_grad += torch.einsum("...ri,ri->...r",
                                          torch.matmul(grad_basis2, dmdmt),
                                          grad_basis2)
                # pytorch's "...ij,ir,jr->...r" is really slow for large matrix
                # grad_grad = torch.einsum("...ij,ir,jr->...r", dmdmt, self.grad_basis[0], self.grad_basis[0])
                # grad_grad += torch.einsum("...ij,ir,jr->...r", dmdmt, self.grad_basis[1], self.grad_basis[1])
                # grad_grad += torch.einsum("...ij,ir,jr->...r", dmdmt, self.grad_basis[2], self.grad_basis[2])
                lapldens[..., ioff:iend] = (lapl_basis + grad_grad) * 2
                kindens[..., ioff:iend] = grad_grad * 0.5

        # dens: (*BD, ngrid)
        # gdens: (*BD, ndim, ngrid)
        res = ValGrad(value=dens, grad=gdens, lapl=lapldens, kin=kindens)
        return res

    def _get_vxc_from_potinfo(self, potinfo: ValGrad) -> xt.LinearOperator:
        # obtain the vxc operator from the potential information
        # potinfo.value: (*BD, nr)
        # potinfo.grad: (*BD, ndim, nr)
        # potinfo.lapl: (*BD, nr)
        # potinfo.kin: (*BD, nr)
        # self.basis: (nr, nao)
        # self.grad_basis: (ndim, nr, nao)

        # prepare the fock matrix component from vxc
        nao = self.basis.shape[-1]
        mat = torch.zeros((*potinfo.value.shape[:-1], nao, nao),
                          dtype=self.dtype,
                          device="cuda:0")
        potinfo.value = potinfo.value.to("cuda:0")
        # Split the r-dimension into several parts, it is usually faster than
        # evaluating all at once
        maxnumel = config.CHUNK_MEMORY // get_dtype_memsize(self.basis)
        for basis, ioff, iend in chunkify(self.basis, dim=0, maxnumel=maxnumel):
            # basis: (nr, nao)
            vb = potinfo.value[..., ioff:iend].unsqueeze(
                -1) * basis  # (*BD, nr, nao)
            if self.xcfamily in [2, 4]:  # GGA or MGGA
                assert potinfo.grad is not None  # (..., ndim, nr)
                vgrad = potinfo.grad[..., ioff:iend] * 2
                grad_basis0 = self.grad_basis[0, ioff:iend, :]  # (nr, nao)
                grad_basis1 = self.grad_basis[1, ioff:iend, :]
                grad_basis2 = self.grad_basis[2, ioff:iend, :]
                vb += torch.einsum("...r,ra->...ra", vgrad[..., 0, :],
                                   grad_basis0)
                vb += torch.einsum("...r,ra->...ra", vgrad[..., 1, :],
                                   grad_basis1)
                vb += torch.einsum("...r,ra->...ra", vgrad[..., 2, :],
                                   grad_basis2)
            if self.xcfamily == 4:  # MGGA
                assert potinfo.lapl is not None  # (..., nrgrid)
                assert potinfo.kin is not None
                lapl = potinfo.lapl[..., ioff:iend]
                kin = potinfo.kin[..., ioff:iend]
                vb += 2 * lapl.unsqueeze(-1) * self.lapl_basis[ioff:iend, :]

            # calculating the matrix from multiplication with the basis
            mat += torch.matmul(
                self.basis_dvolume[ioff:iend, :].transpose(-2, -1), vb)

            if self.xcfamily == 4:  # MGGA
                assert potinfo.lapl is not None  # (..., nrgrid)
                assert potinfo.kin is not None
                lapl_kin_dvol = (2 * lapl + 0.5 * kin) * self.dvolume[...,
                                                                      ioff:iend]
                mat += torch.einsum("...r,rb,rc->...bc", lapl_kin_dvol,
                                    grad_basis0, grad_basis0)
                mat += torch.einsum("...r,rb,rc->...bc", lapl_kin_dvol,
                                    grad_basis1, grad_basis1)
                mat += torch.einsum("...r,rb,rc->...bc", lapl_kin_dvol,
                                    grad_basis2, grad_basis2)

        # construct the Hermitian linear operator
        mat = self._orthozer.convert2(mat)
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        vxc_linop = xt.LinearOperator.m(mat, is_hermitian=True)
        return vxc_linop


class Mol(dqc.system.mol.Mol):

    def __init__(
            self,
            moldesc: Union[str, Tuple[AtomZsType, AtomPosType]],
            basis: BasisInpType,
            *,
            orthogonalize_basis: bool = True,
            ao_parameterizer: str = "qr",
            grid: Union[int, str] = "sg3",
            spin: Optional[ZType] = None,
            charge: ZType = 0,
            orb_weights: Optional[SpinParam[torch.Tensor]] = None,
            efield: Union[torch.Tensor, Tuple[torch.Tensor, ...], None] = None,
            vext: Optional[torch.Tensor] = None,
            dtype: torch.dtype = torch.float64,
            device: torch.device = torch.device('cpu'),
    ):
        self._dtype = dtype
        self._device = device
        self._grid_inp = grid
        self._basis_inp = basis
        self._grid: Optional[BaseGrid] = None
        self._vext = vext

        # make efield a tuple
        self._efield = dqc.system.mol._normalize_efield(efield)
        self._preproc_efield = dqc.system.mol._preprocess_efield(self._efield)

        # initialize cache
        self._cache = Cache()

        # get the AtomCGTOBasis & the hamiltonian
        # atomzs: (natoms,) dtype: torch.int or dtype for floating point
        # atompos: (natoms, ndim)
        atomzs, atompos = parse_moldesc(moldesc, dtype=dtype, device=device)
        atomzs_int = torch.round(atomzs).to(
            torch.int) if atomzs.is_floating_point() else atomzs
        allbases = dqc.system.mol._parse_basis(
            atomzs_int, basis)  # list of list of CGTOBasis
        atombases = [
            AtomCGTOBasis(atomz=atz, bases=bas, pos=atpos)
            for (atz, bas, atpos) in zip(atomzs, allbases, atompos)
        ]
        self._atombases = atombases
        self._hamilton = Hamiltonz(atombases,
                                   efield=self._preproc_efield,
                                   vext=self._vext,
                                   cache=self._cache.add_prefix("hamilton"),
                                   orthozer=orthogonalize_basis,
                                   aoparamzer=ao_parameterizer)
        self._orthogonalize_basis = orthogonalize_basis
        self._aoparamzer = ao_parameterizer
        self._atompos = atompos  # (natoms, ndim)
        self._atomzs = atomzs  # (natoms,) int-type or dtype if floating point
        self._atomzs_int = atomzs_int  # (natoms,) int-type rounded from atomzs
        nelecs_tot: torch.Tensor = torch.sum(atomzs)

        # orb_weights is not specified, so determine it from spin and charge
        if orb_weights is None:
            # get the number of electrons and spin
            nelecs, spin, frac_mode = dqc.system.mol._get_nelecs_spin(
                nelecs_tot, spin, charge)
            _orb_weights, _orb_weights_u, _orb_weights_d = dqc.system.mol._get_orb_weights(
                nelecs, spin, frac_mode, dtype, device)

            # save the system's properties
            self._spin = spin
            self._charge = charge
            self._numel = nelecs
            self._orb_weights = _orb_weights
            self._orb_weights_u = _orb_weights_u
            self._orb_weights_d = _orb_weights_d

        # orb_weights is specified, so calculate the spin and charge from it
        else:
            if not isinstance(orb_weights, SpinParam):
                raise TypeError(
                    "Specifying orb_weights must be in SpinParam type")
            assert orb_weights.u.ndim == 1
            assert orb_weights.d.ndim == 1
            assert len(orb_weights.u) == len(orb_weights.d)

            # check if it is decreasing
            orb_u_dec = torch.all(orb_weights.u[:-1] -
                                  orb_weights.u[1:] > -1e-4)
            orb_d_dec = torch.all(orb_weights.d[:-1] -
                                  orb_weights.d[1:] > -1e-4)
            if not (orb_u_dec and orb_d_dec):
                # if not decreasing, the variational might give the wrong results
                warnings.warn(
                    "The orbitals should be ordered in a non-increasing manner. "
                    "Otherwise, some calculations might be wrong.")

            utot = orb_weights.u.sum()
            dtot = orb_weights.d.sum()
            self._numel = utot + dtot
            self._spin = utot - dtot
            self._charge = nelecs_tot - self._numel

            self._orb_weights_u = orb_weights.u
            self._orb_weights_d = orb_weights.d
            self._orb_weights = orb_weights.u + orb_weights.d


class DFTSystem():
    """
    The DFTSystem class creates and returns the various systems in an entry object as dictionaries.

    Examples
    --------
    >>> from deepchem.feat.dft_data import DFTSystem
    >>> systems = {'moldesc': 'Li 1.5070 0 0; H -1.5070 0 0','basis': '6-311++G(3df,3pd)'}
    >>> output = DFTSystem(systems)

    Returns
    -------
    DFTSystem object for all the individual atoms/ions/molecules in an entry object.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.

    https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/system/base_system.py
    """

    def __init__(self, system: Dict):
        self.system = system
        self.moldesc = system["moldesc"]
        self.basis = system["basis"]
        self.spin = 0
        self.charge = 0
        self.no = 1
        if 'spin' in system.keys():
            self.spin = int(system["spin"])
        if 'charge' in system.keys():
            self.charge = int(system["charge"])
        if 'number' in system.keys():
            self.no = int(system["number"])
        """
        Parameters
        ----------
        system: Dict
            system is a dictionary containing information on the atomic positions,
            atomic numbers, and basis set used for the DFT calculation.
        """

    def get_dqc_mol(self, pos_reqgrad: bool = False) -> BaseSystem:
        """
        This method converts the system dictionary to a DQC system and returns it.
        Parameters
        ----------
        pos_reqgrad: bool
            decides if the atomic position require gradient calculation.
        Returns
        -------
        mol
            DQC mol object
        """
        atomzs, atomposs = dqc.parse_moldesc(self.moldesc)
        if pos_reqgrad:
            atomposs.requires_grad_()
        mol = Mol(self.moldesc,
                  self.basis,
                  spin=self.spin,
                  charge=self.charge,
                  device="cuda:0")
        print(mol)
        return mol


class DFTEntry():
    """
    Handles creating and initialising DFTEntry objects from the dataset. This object contains information    about the various systems in the datapoint (atoms, molecules and ions) along with the ground truth
    values.
    Notes
    -----
    Entry class should not be initialized directly, but created through
    ``Entry.create``

    Example
    -------
    >>> from deepchem.feat.dft_data import DFTEntry
    >>> e_type= 'dm'
    >>> true_val= 'deepchem/data/tests/dftHF_output.npy'
    >>> systems = [{'moldesc': 'H 0.86625 0 0; F -0.86625 0 0','basis': '6-311++G(3df,3pd)'}]
    >>> dm_entry_for_HF = DFTEntry.create(e_type, true_val, systems)
    """

    @classmethod
    def create(self,
               e_type: str,
               true_val: Optional[str],
               systems: List[Dict],
               weight: Optional[int] = 1):
        """
        This method is used to initialise the DFTEntry class. The entry objects are created
        based on their entry type.

        Parameters
        ----------
        e_type: str
            Determines the type of calculation to be carried out on the entry
            object. Accepts the following values: "ae", "ie", "dm", "dens", that stand for atomization energy,
            ionization energy, density matrix and density profile respectively.
        true_val: str
            Ground state energy values for the entry object as a string (for ae
            and ie), or a .npy file containing a matrix ( for dm and dens).
        systems: List[Dict]
            List of dictionaries contains "moldesc", "basis" and "spin"
            of all the atoms/molecules. These values are to be entered in
            the DQC or PYSCF format. The systems needs to be entered in a
            specific order, i.e ; the main atom/molecule needs to be the
            first element. (This is for objects containing equations, such
            as ae and ie entry objects). Spin and charge of the system are
            optional parameters and are considered '0' if not specified.
            The system number refers to the number of times the systems is
            present in the molecule - this is for polyatomic molecules and the
            default value is 1. For example ; system number of Hydrogen in water
            is 2.
        weight: int
            Weight of the entry object.
        Returns
        -------
        DFTEntry object based on entry type

        """
        if true_val is None:
            true_val = '0.0'
        if e_type == "ae":
            return _EntryAE(e_type, true_val, systems, weight)
        elif e_type == "ie":
            return _EntryIE(e_type, true_val, systems, weight)
        elif e_type == "dm":
            return _EntryDM(e_type, true_val, systems, weight)
        elif e_type == "dens":
            return _EntryDens(e_type, true_val, systems, weight)
        else:
            raise NotImplementedError("Unknown entry type: %s" % e_type)

    def __init__(self,
                 e_type: str,
                 true_val: Optional[str],
                 systems: List[Dict],
                 weight: Optional[int] = 1):
        self._systems = [DFTSystem(p) for p in systems]
        self._weight = weight

    def get_systems(self) -> List[DFTSystem]:
        """
        Parameters
        ----------
        systems:  List[DFTSystem]

        Returns
        -------
        List of systems in the entry
        """
        return self._systems

    @abstractproperty
    def entry_type(self) -> str:
        """
        Returns
        -------
        The type of entry ;
        1) Atomic Ionization Potential (IP/IE)
        2) Atomization Energy (AE)
        3) Density Profile (DENS)
        4) Density Matrix (DM)
        """
        pass

    def get_true_val(self) -> np.ndarray:
        """
        Get the true value of the DFTEntry.
        For the AE and IP entry types, the experimental values are collected from the NIST CCCBDB/ASD
        databases.
        The true values of density profiles are calculated using PYSCF-CCSD calculations. This method            simply loads the value, no calculation is performed.
        """
        return np.array(0)

    @abstractmethod
    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        """
        Return the energy value of the entry, using a DQC-DFT calculation, where the XC has been
        replaced by the trained neural network. This method does not carry out any calculations, it is
        an interface to the KSCalc utility.
        """
        pass

    def get_weight(self):
        """
        Returns
        -------
        Weight of the entry object
        """
        return self._weight


class _EntryDM(DFTEntry):
    """
    Entry for Density Matrix (DM)
    The density matrix is used to express total energy of both non-interacting and
    interacting systems.

    Notes
    -----
    dm entry can only have 1 system
    """

    def __init__(self, e_type, true_val, systems, weight):
        """
        Parameters
        ----------
        e_type: str
        true_val: str
           must be a .npy file containing the pre-calculated density matrix
        systems: List[Dict]

        """
        super().__init__(e_type, true_val, systems)
        self.true_val = true_val
        assert len(self.get_systems()) == 1
        self._weight = weight

    @property
    def entry_type(self) -> str:
        return "dm"

    def get_true_val(self) -> np.ndarray:
        # get the density matrix from PySCF's CCSD calculation
        dm = np.load(self.true_val)
        return dm

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        val = qcs[0].aodmtot()
        return np.array([val.tolist()])


class _EntryDens(DFTEntry):
    """
    Entry for density profile (dens), compared with CCSD calculation
    """

    def __init__(self, e_type, true_val, systems, weight):
        """
        Parameters
        ----------
        e_type: str
        true_val: str
           must be a .npy file containing the pre-calculated density profile.
        systems: List[Dict]
        """
        super().__init__(e_type, true_val, systems)
        self.true_val = true_val
        assert len(self.get_systems()) == 1
        self._grid: Optional[BaseGrid] = None
        self._weight = weight

    @property
    def entry_type(self) -> str:
        return "dens"

    def get_true_val(self) -> np.ndarray:
        dens = np.load(self.true_val)
        return dens

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        """
        This method calculates the integration grid which is then used to calculate the
        density profile of an entry object.

        Parameters
        ----------
        qcs: List[KSCalc]

        Returns
        -------
        Numpy array containing calculated density profile
        """
        qc = qcs[0]

        grid = self.get_integration_grid()
        rgrid = grid.get_rgrid()
        val = qc.dens(rgrid)
        return np.array(val.tolist())

    def get_integration_grid(self) -> BaseGrid:
        """
        This method is used to calculate the integration grid required for a system
        in order to calculate it's density profile using Differentiable DFT.

        Returns
        -------
        grid: BaseGrid

        References
        ----------
        https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/grid/base_grid.py
        """
        if self._grid is None:
            system = self.get_systems()[0]
            dqc_mol = system.get_dqc_mol()
            dqc_mol.setup_grid()
            grid = dqc_mol.get_grid()
            assert grid.coord_type == "cart"
            self._grid = grid

        return self._grid


class _EntryIE(DFTEntry):
    """
    Entry for Ionization Energy (IE)
    """

    def __init__(self, e_type, true_val, systems, weight):
        """
        Parameters
        ----------
        e_type: str
        true_val: str
        systems: List[Dict]
        """
        super().__init__(e_type, true_val, systems)
        self.true_val = float(true_val)
        self._weight = weight

    @property
    def entry_type(self) -> str:
        return "ie"

    def get_true_val(self) -> np.ndarray:
        return np.array([self.true_val])

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        """
        This method calculates the energy of an entry based on the systems and command present
        in the data object. For example; for a Lithium hydride molecule the total energy
        would be ; energy(Li) + energy(H) - energy(LiH)

        Parameters
        ----------
        qcs: List[KSCalc]

        Returns
        -------
        Total Energy of a data object for entry types IE and AE
        """
        systems = [i.no for i in self.get_systems()]
        e_1 = [m.energy() for m in qcs]
        e = [item1 * item2 for item1, item2 in zip(systems, e_1)]
        val = sum(e) - 2 * e[0]
        return np.array([val.tolist()])


class _EntryAE(_EntryIE):
    """
    Entry for Atomization Energy (AE)
    """

    @property
    def entry_type(self) -> str:
        return "ae"
