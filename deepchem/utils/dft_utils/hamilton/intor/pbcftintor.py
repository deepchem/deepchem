from typing import Tuple, Optional, List
import warnings
import ctypes
import operator
from functools import reduce
import torch
import numpy as np
from deepchem.utils.dft_utils import LibcintWrapper, Lattice
from deepchem.utils.dft_utils.hamilton.intor.utils import np2ctypes, int2ctypes, CGTO, CPBC, c_null_ptr
from deepchem.utils.dft_utils.hamilton.intor.pbcintor import PBCIntOption, _check_and_set_pbc, \
    get_default_options, get_default_kpts, _concat_atm_bas_env
from deepchem.utils.pytorch_utils import get_complex_dtype, estimate_ovlp_rcut
from deepchem.utils.dft_utils.hamilton.intor.namemgr import IntorNameManager


# Fourier transform integrals
def pbcft_overlap(wrapper: LibcintWrapper,
                  shortname: str = 'ovlp',
                  other: Optional[LibcintWrapper] = None,
                  gvgrid: Optional[torch.Tensor] = None,
                  kpts: Optional[torch.Tensor] = None,
                  options: Optional[PBCIntOption] = None):
    r"""Compute PBC Fourier transform overlap integrals.

    Performing the periodic boundary condition (PBC) on 1-electron Fourier
    Transform integrals, i.e.

    $$
    \sum_\mathbf{T} e^{-i \mathbf{k}\cdot\mathbf{T}} \int \exp(-i\mathbf{G}\cdot\mathbf{r})
    \phi_i(\mathbf{r}) \phi_j(\mathbf{r}-\mathbf{T})\ \mathrm{d}\mathbf{r}
    $$

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, loadbasis, Lattice
    >>> from deepchem.utils.dft_utils.hamilton.intor.pbcftintor import pbcft_overlap
    >>> # Create a simple lattice
    >>> a = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    >>> lattice = Lattice(a)
    >>> # Create atom with basis
    >>> pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    >>> basis = loadbasis("1:STO-3G", dtype=torch.float64, requires_grad=False)
    >>> atom = AtomCGTOBasis(atomz=1, bases=basis, pos=pos)
    >>> wrapper = LibcintWrapper([atom], spherical=True, lattice=lattice)
    >>> # Create G-grid
    >>> gvgrid = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=torch.float64)
    >>> # Compute PBC FT overlap integral
    >>> result = pbcft_overlap(wrapper, gvgrid=gvgrid)
    >>> result.shape
    torch.Size([1, 1, 1, 2])

    Parameters
    ----------
    wrapper : LibcintWrapper
        The environment wrapper containing the basis.
    other : Optional[LibcintWrapper]
        Another environment wrapper containing the basis. If unspecified,
        then `other = wrapper`.
    gvgrid : Optional[torch.Tensor]
        The reciprocal coordinate of G with shape `(nggrid, ndim)`.
        If unspecified, then it is assumed to be all zeros.
    kpts : Optional[torch.Tensor]
        k-points where the integration is supposed to be performed.
        If specified, it should have the shape of `(nkpts, ndim)`.
        Otherwise, it is assumed to be all zeros.
    options : Optional[PBCIntOption]
        The integration options. If unspecified, uses the default
        value of `PBCIntOption`.

    Returns
    -------
    torch.Tensor
        A complex tensor representing the overlap integral with shape
        `(nkpts, *ncomp, nwrapper, nother, nggrid)`.

    """
    # check and set the default values
    other1 = _check_and_set_pbc(wrapper, other)
    options1 = get_default_options(options)
    kpts1 = get_default_kpts(kpts, dtype=wrapper.dtype, device=wrapper.device)
    gvgrid1 = get_default_kpts(gvgrid,
                               dtype=wrapper.dtype,
                               device=wrapper.device)

    assert isinstance(wrapper.lattice, Lattice)
    return _PBCInt2cFTFunction.apply(*wrapper.params, *wrapper.lattice.params,
                                     gvgrid1, kpts1, [wrapper, other1],
                                     IntorNameManager("int1e",
                                                      shortname), options1)


# torch autograd function wrappers
class _PBCInt2cFTFunction(torch.autograd.Function):
    """Autograd wrapper for PBC 2-center Fourier transform integrals.

    This class wraps the periodic boundary condition 2-center Fourier transform
    integral computation as a PyTorch autograd function. It handles the forward
    pass computation and provides a stub for the backward pass.

    Note
    ----
    Gradients are not implemented for this function and will raise
    NotImplementedError if called.

    """
    # wrapper class for the periodic boundary condition 2-centre integrals
    @staticmethod
    def forward(
            ctx,  # type: ignore
            allcoeffs: torch.Tensor,
            allalphas: torch.Tensor,
            allposs: torch.Tensor,
            alattice: torch.Tensor,
            gvgrid: torch.Tensor,
            kpts: torch.Tensor,
            wrappers: List[LibcintWrapper],
            int_nmgr: IntorNameManager,
            options: PBCIntOption) -> torch.Tensor:
        """Forward pass for PBC 2-center FT integrals.

        Parameters
        ----------
        ctx : Context
            Context object for saving tensors for backward pass.
        allcoeffs : torch.Tensor
            Gaussian contraction coefficients, shape `(ngauss_tot,)`.
        allalphas : torch.Tensor
            Gaussian exponents, shape `(ngauss_tot,)`.
        allposs : torch.Tensor
            Atomic positions, shape `(natom, ndim)`.
        alattice : torch.Tensor
            Lattice vectors, shape `(ndim, ndim)`.
        gvgrid : torch.Tensor
            G-space grid coordinates, shape `(nggrid, ndim)`.
        kpts : torch.Tensor
            k-points, shape `(nkpts, ndim)`.
        wrappers : List[LibcintWrapper]
            List of LibcintWrapper objects containing basis info.
        int_nmgr : IntorNameManager
            Integral name manager.
        options : PBCIntOption
            Integration options.

        Returns
        -------
        torch.Tensor
            Computed integral tensor.

        """
        out_tensor = PBCFTIntor(int_nmgr, wrappers, gvgrid, kpts,
                                options).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs, alattice, gvgrid,
                              kpts)
        ctx.other_info = (wrappers, int_nmgr, options)
        return out_tensor

    @staticmethod
    def backward(
        ctx, *grad_out: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Backward pass for PBC 2-center FT integrals.

        Parameters
        ----------
        ctx: Context
            Context object with saved tensors.
        grad_out: torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[Optional[torch.Tensor], ...]
            Not implemented, raises NotImplementedError.

        Raises
        ------
        NotImplementedError
            Gradients are not supported for PBC FT integrals.

        """
        raise NotImplementedError(
            "gradients of PBC 2-centre FT integrals are not implemented")


# integrator object (direct interface to lib*)
class PBCFTIntor(object):
    """Integrator class for PBC Fourier transform integrals.

    This class computes periodic boundary condition 2-center integrals using
    Fourier transform methods. It interfaces with the libcint library to
    perform the actual computations.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, loadbasis, Lattice
    >>> from deepchem.utils.dft_utils.hamilton.intor.pbcftintor import PBCFTIntor
    >>> from deepchem.utils.dft_utils.hamilton.intor.namemgr import IntorNameManager
    >>> from deepchem.utils.dft_utils.hamilton.intor.pbcintor import PBCIntOption
    >>> # Create lattice and wrappers
    >>> a = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    >>> lattice = Lattice(a)
    >>> pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    >>> basis = loadbasis("1:STO-3G", dtype=torch.float64, requires_grad=False)
    >>> atoms = [AtomCGTOBasis(atomz=1, bases=basis, pos=pos[i]) for i in range(2)]
    >>> wrappers = [LibcintWrapper([atoms[i]], spherical=True, lattice=lattice) for i in range(2)]
    >>> # Create integrator
    >>> gvgrid = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
    >>> kpts = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    >>> options = PBCIntOption()
    >>> int_nmgr = IntorNameManager("int1e", "ovlp")
    >>> intor = PBCFTIntor(int_nmgr, wrappers, gvgrid, kpts, options)
    >>> result = intor.calc()
    >>> result.shape
    torch.Size([1, 1, 1, 1])

    Note
    ----
    This class is designed for single-use (one-time integral computation).
    Calling `calc()` more than once will raise an assertion error.

    """

    def __init__(self, int_nmgr: IntorNameManager,
                 wrappers: List[LibcintWrapper], gvgrid_inp: torch.Tensor,
                 kpts_inp: torch.Tensor, options: PBCIntOption):
        """Initialize the PBC FT integrator.

        Parameters
        ----------
        int_nmgr : IntorNameManager
            Integral name manager for specifying the integral type.
        wrappers : List[LibcintWrapper]
            List of LibcintWrapper objects containing basis function data.
        gvgrid_inp : torch.Tensor
            G-space grid coordinates, shape `(nggrid, ndim)`.
        kpts_inp : torch.Tensor
            k-points for the integration, shape `(nkpts, ndim)`.
        options : PBCIntOption
            Integration options including precision and screening settings.

        """
        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        kpts_inp_np = kpts_inp.detach().cpu().numpy()  # (nk, ndim)
        GvT = np.asarray(gvgrid_inp.detach().cpu().numpy().T,
                         order="C")  # (ng, ndim)
        opname = int_nmgr.get_ft_intgl_name(wrapper0.spherical)
        lattice = wrapper0.lattice
        assert isinstance(lattice, Lattice)

        # get the output's component shape
        comp_shape = int_nmgr.get_intgl_components_shape()
        ncomp = reduce(operator.mul, comp_shape, 1)

        # estimate the rcut and the lattice translation vectors
        coeffs, alphas, _ = wrapper0.params
        rcut = estimate_ovlp_rcut(options.precision, coeffs, alphas)
        ls = np.asarray(lattice.get_lattice_ls(rcut=rcut).cpu())

        self.int_type = int_nmgr.int_type
        self.wrappers = wrappers
        self.GvT = GvT
        self.kpts_inp_np = kpts_inp_np
        self.opname = opname
        self.dtype = wrapper0.dtype
        self.device = wrapper0.device
        self.comp_shape = comp_shape
        self.ncomp = ncomp
        self.ls = ls
        self.options = options

        # this class is meant to be used once
        self.integral_done = False

    def calc(self) -> torch.Tensor:
        """Compute the PBC Fourier transform integral.

        Returns
        -------
        torch.Tensor
            The computed integral tensor.

        Raises
        ------
        AssertionError
            If the integral has already been computed (single-use only).
        ValueError
            If the integral type is unknown.

        """
        assert not self.integral_done
        self.integral_done = True
        if self.int_type == "int1e":
            return self._int2c()
        else:
            raise ValueError("Unknown integral type: %s" % self.int_type)

    def _int2c(self) -> torch.Tensor:
        """Compute 2-center PBC Fourier transform integral.

        This method implements the 2-center integral computation using
        lattice summation and Fourier transform, replicating the
        `ft_aopair_kpts` function from pyscf [1].

        Returns
        -------
        torch.Tensor
            Computed 2-center integral tensor.

        Reference
        ---------
        [1].. https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/df/ft_ao.py

        """

        assert len(self.wrappers) == 2

        # if the ls is too big, it might produce segfault
        if (self.ls.shape[0] > 1e6):
            warnings.warn(
                "The number of neighbors in the integral is too many, "
                "it might causes segfault")

        # libpbc will do in-place shift of the basis of one of the wrappers, so
        # we need to make a concatenated copy of the wrapper's atm_bas_env
        atm, bas, env, ao_loc = _concat_atm_bas_env(self.wrappers[0],
                                                    self.wrappers[1])
        i0, i1 = self.wrappers[0].shell_idxs
        j0, j1 = self.wrappers[1].shell_idxs
        nshls0 = len(self.wrappers[0].parent)
        shls_slice = (i0, i1, j0 + nshls0, j1 + nshls0)

        # get the lattice translation vectors and the exponential factors
        expkl = np.asarray(np.exp(1j * np.dot(self.kpts_inp_np, self.ls.T)),
                           order='C')

        # prepare the output
        nGv = self.GvT.shape[-1]
        nkpts = len(self.kpts_inp_np)
        outshape = (nkpts,) + self.comp_shape + tuple(
            w.nao() for w in self.wrappers) + (nGv,)
        out = np.empty(outshape, dtype=np.complex128)

        # do the integration
        cintor = getattr(CGTO(), self.opname)
        eval_gz = CPBC().GTO_Gv_general
        fill = CPBC().PBC_ft_fill_ks1
        drv = CPBC().PBC_ft_latsum_drv
        p_gxyzT = c_null_ptr()
        p_mesh = (ctypes.c_int * 3)(0, 0, 0)
        p_b = (ctypes.c_double * 1)(0)
        drv(cintor, eval_gz, fill, np2ctypes(out), int2ctypes(nkpts),
            int2ctypes(self.ncomp), int2ctypes(len(self.ls)),
            np2ctypes(self.ls), np2ctypes(expkl),
            (ctypes.c_int * len(shls_slice))(*shls_slice), np2ctypes(ao_loc),
            np2ctypes(self.GvT), p_b, p_gxyzT, p_mesh, int2ctypes(nGv),
            np2ctypes(atm), int2ctypes(len(atm)), np2ctypes(bas),
            int2ctypes(len(bas)), np2ctypes(env))

        out_tensor = torch.as_tensor(out,
                                     dtype=get_complex_dtype(self.dtype),
                                     device=self.device)
        return out_tensor
