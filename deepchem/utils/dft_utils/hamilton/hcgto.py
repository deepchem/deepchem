import logging
import torch
from typing import List, Optional, Union, Tuple
from deepchem.utils import chunkify, get_dtype_memsize
from deepchem.utils.dft_utils import BaseDF, DFMol, DensityFitInfo, eval_gto, eval_gradgto, eval_laplgto, LibcintWrapper, BaseHamilton, overlap, kinetic, nuclattr, int1e, elrep, BaseXC, BaseGrid, AtomCGTOBasis, ValGrad, SpinParam
from deepchem.utils.differentiation_utils import LinearOperator
from deepchem.utils.cache_utils import Cache
from deepchem.utils.dft_utils.hamilton.orbconverter import OrbitalOrthogonalizer

logger = logging.getLogger(__name__)


class HamiltonCGTO(BaseHamilton):
    """
    Hamiltonian object of contracted Gaussian type-orbital. This class
    orthogonalizes the basis by taking the weighted eigenvectors of the
    overlap matrix, i.e. the eigenvectors divided by square root of the
    eigenvalues. The advantage of doing this is making the overlap matrix
    in Roothan's equation identity and it could handle overcomplete basis.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, CGTOBasis, HamiltonCGTO
    >>> cgto = CGTOBasis(angmom=0, alphas=torch.ones(1), coeffs=torch.ones(1))
    >>> atomcgto = AtomCGTOBasis(atomz=1, bases=[cgto], pos=[[0.0, 0.0, 0.0]])

    """

    def __init__(self,
                 atombases: List[AtomCGTOBasis],
                 spherical: bool = True,
                 df: Optional[DensityFitInfo] = None,
                 efield: Optional[Tuple[torch.Tensor, ...]] = None,
                 cache: Optional[Cache] = None) -> None:
        """Initialise the HamiltonCGTO

        Parameters
        ----------
        atombases: List[AtomCGTOBasis]
            Basis information of the concerned atom.
        spherical: bool
            Is the given orbital spherical.
        df: Optional[DensityFitInfo]
            Density Fitting information of the atom.
        efield: Optional[Tuple[torch.Tensor, ...]]
            Electrostatic Force field information about the environment.
        cache: Optional[Cache]
            Optional Cache parameters.

        """
        self.atombases = atombases
        self.spherical = spherical
        self.libcint_wrapper = LibcintWrapper(atombases, spherical)
        self._orthozer = OrbitalOrthogonalizer(overlap(self.libcint_wrapper))
        self.dtype = self.libcint_wrapper.dtype
        self.device = self.libcint_wrapper.device
        self._dfoptions = df
        if df is None:
            self._df: Optional[DFMol] = None
        else:
            self._df = DFMol(df,
                             wrapper=self.libcint_wrapper,
                             orthozer=self._orthozer)

        self._efield = efield
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

    @property
    def nao(self) -> int:
        """Number of Atomic Orbitals.

        Returns
        -------
        int
            No. of Atomic Orbitals.

        """
        return self._orthozer.nao()

    @property
    def kpts(self) -> torch.Tensor:
        """The KPOINTS specifies the Bloch vectors (k points) used to
        sample the Brillouin zone. Converging this sampling is one of
        the essential tasks in many calculations concerning the
        electronic minimization.

        Bloch sphere is a geometrical representation of the pure state
        space of a two-level quantum mechanical system (qubit).

        Returns
        -------
        torch.Tensor
            List of k-points in the Hamiltonian. Shape: (nkpts, ndim)

        """
        raise TypeError(
            "Isolated molecule Hamiltonian does not have kpts property")

    @property
    def df(self) -> Optional[BaseDF]:
        """
        Returns the density fitting object (if any) attached to this
        Hamiltonian object.

        Returns
        -------
        Optional[BaseDF]
            Returns the density fitting object (if any) attached to this
            Hamiltonian object.

        """
        return self._df

    # setups
    def build(self) -> BaseHamilton:
        """
        Construct the elements needed for the Hamiltonian.
        Heavy-lifting operations should be put here.

        Returns
        -------
        BaseHamilton
            The Hamiltonian representing the total energy operator for a system of
            interacting electrons.
        """
        # get the matrices (all (nao, nao), except el_mat)
        # these matrices have already been normalized
        with self._cache.open():

            # check the signature
            self._cache.check_signature({
                "atombases": self.atombases,
                "spherical": self.spherical,
                "dfoptions": self._dfoptions,
            })

            logger.info("Calculating the overlap matrix")
            self.olp_mat = self._cache.cache(
                "overlap", lambda: overlap(self.libcint_wrapper))
            logger.info("Calculating the kinetic matrix")
            kin_mat = self._cache.cache("kinetic",
                                        lambda: kinetic(self.libcint_wrapper))
            logger.info("Calculating the nuclear attraction matrix")
            nucl_mat = self._cache.cache("nuclattr",
                                         lambda: nuclattr(self.libcint_wrapper))
            self.nucl_mat = nucl_mat
            self.kinnucl_mat = kin_mat + nucl_mat

            # electric field integral
            if self._efield is not None:
                # (ndim, nao, nao)
                fac: float = 1.0
                for i in range(len(self._efield)):
                    fac *= i + 1

                    def intor_fcn():
                        return int1e("r0" * (i + 1), self.libcint_wrapper)

                    efield_mat_f = self._cache.cache(f"efield{i}", intor_fcn)
                    efield_mat = torch.einsum("dab,d->ab", efield_mat_f,
                                              self._efield[i])
                    self.kinnucl_mat = self.kinnucl_mat + efield_mat / fac

            if self._df is None:
                logger.info("Calculating the electron repulsion matrix")
                self.el_mat = self._cache.cache(
                    "elrep", lambda: elrep(self.libcint_wrapper))  # (nao^4)
                # TODO: decide whether to precompute the 2-eris in the new basis
                # based on the memory
                self.el_mat = self._orthozer.convert4(self.el_mat)
            else:
                logger.info("Building the density fitting matrices")
                self._df.build()
            self.is_built = True

            # orthogonalize the matrices
            self.olp_mat = self._orthozer.convert2(self.olp_mat)  # (nao2, nao2)
            self.kinnucl_mat = self._orthozer.convert2(self.kinnucl_mat)
            self.nucl_mat = self._orthozer.convert2(self.nucl_mat)

            logger.info("Setting up the Hamiltonian done")

        return self

    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        """
        Setup the basis (with its grad) in the spatial grid and prepare the
        gradient of atomic orbital according to the ones required by the xc.
        If xc is not given, then only setup the grid with ao (without any
        gradients of ao)

        Parameters
        ----------
        grid: BaseGrid
            Grid used to setup this Hamilton.
        xc: Optional[BaseXC] (default None)
            Exchange Corelation functional of this Hamiltonian.

        """
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
        logger.info("Calculating the basis values in the grid")
        self.is_ao_set = True
        self.basis = eval_gto(self.libcint_wrapper,
                              self.rgrid,
                              to_transpose=True)  # (ngrid, nao)
        self.dvolume = self.grid.get_dvolume()
        self.basis_dvolume = self.basis * self.dvolume.unsqueeze(
            -1)  # (ngrid, nao)

        if self.xcfamily == 1:  # LDA
            return

        # setup the gradient of the basis
        logger.info("Calculating the basis gradient values in the grid")
        self.is_grad_ao_set = True
        # (ndim, nao, ngrid)
        self.grad_basis = eval_gradgto(self.libcint_wrapper,
                                       self.rgrid,
                                       to_transpose=True)
        if self.xcfamily == 2:  # GGA
            return

        # setup the laplacian of the basis
        self.is_lapl_ao_set = True
        logger.info("Calculating the basis laplacian values in the grid")
        self.lapl_basis = eval_laplgto(self.libcint_wrapper,
                                       self.rgrid,
                                       to_transpose=True)  # (nao, ngrid)

    # fock matrix components
    def get_nuclattr(self) -> LinearOperator:
        """LinearOperator of the nuclear Coulomb attraction.

        Nuclear Coulomb attraction is the electrostatic force binding electrons
        to a nucleus. Positively charged protons attract negatively charged
        electrons, creating stability in quantum systems. This force plays a
        fundamental role in determining the structure and behavior of atoms,
        contributing significantly to the overall potential energy in atomic
        physics.

        Returns
        -------
        LinearOperator
            LinearOperator of the nuclear Coulomb attraction. Shape: (`*BH`, nao, nao)

        """
        return LinearOperator.m(self.nucl_mat, is_hermitian=True)

    def get_kinnucl(self) -> LinearOperator:
        """
        Returns the LinearOperator of the one-electron operator (i.e. kinetic
        and nuclear attraction). Action of a LinearOperator on a function is a
        linear transformation. In the case of one-electron operators, these
        transformations are essential for solving the Schrödinger equation and
        understanding the behavior of electrons in an atomic or molecular system.

        Returns
        -------
        LinearOperator
            LinearOperator of the one-electron operator. Shape: (`*BH`, nao, nao)

        """
        return LinearOperator.m(self.kinnucl_mat, is_hermitian=True)

    def get_overlap(self) -> LinearOperator:
        """
        Returns the LinearOperator representing the overlap of the basis.
        The overlap of the basis refers to the degree to which atomic or
        molecular orbitals in a quantum mechanical system share common space.

        Returns
        -------
        LinearOperator
            LinearOperator representing the overlap of the basis.
            Shape: (`*BH`, nao, nao)

        """
        return LinearOperator.m(self.olp_mat, is_hermitian=True)

    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        """
        Obtains the LinearOperator of the Coulomb electron repulsion operator.
        Known as the J-matrix.

        In the context of electronic structure theory, it accounts for the
        repulsive interaction between electrons in a many-electron system. The
        J-matrix elements involve the Coulombic interactions between pairs of
        electrons, influencing the total energy and behavior of the system.

        Parameters
        ----------
        dm: torch.Tensor
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        LinearOperator
            LinearOperator of the Coulomb electron repulsion operator.
            Shape: (`*BDH`, nao, nao)

        """
        if self._df is None:
            mat = torch.einsum("...ij,ijkl->...kl", dm, self.el_mat)
            mat = (mat +
                   mat.transpose(-2, -1)) * 0.5  # reduce numerical instability
            return LinearOperator.m(mat, is_hermitian=True)
        else:
            elrep = self._df.get_elrep(dm)
            return elrep

    def get_exchange(
        self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
    ) -> Union[LinearOperator, SpinParam[LinearOperator]]:
        """
        Obtains the LinearOperator of the exchange operator.
        It is -0.5 * K where K is the K matrix obtained from 2-electron integral.

        Exchange operator is a mathematical representation of the exchange
        interaction between identical particles, such as electrons. The
        exchange operator quantifies the effect of interchanging the
        positions of two particles.

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        Union[LinearOperator, SpinParam[LinearOperator]]
            LinearOperator of the exchange operator. Shape: (`*BDH`, nao, nao)

        """
        if self._df is not None:
            raise RuntimeError(
                "Exact exchange cannot be computed with density fitting")
        elif isinstance(dm, torch.Tensor):
            # the einsum form below is to hack PyTorch's bug #57121
            # mat = -0.5 * torch.einsum("...jk,ijkl->...il", dm, self.el_mat)  # slower
            mat = -0.5 * torch.einsum("...il,ijkl->...ijk", dm,
                                      self.el_mat).sum(dim=-3)  # faster

            mat = (mat +
                   mat.transpose(-2, -1)) * 0.5  # reduce numerical instability
            return LinearOperator.m(mat, is_hermitian=True)
        else:  # dm is SpinParam
            # using the spin-scaling property of exchange energy
            return SpinParam(
                u=self.get_exchange(2 * dm.u),  # type: ignore
                d=self.get_exchange(2 * dm.d))  # type: ignore

    def get_vext(self, vext: torch.Tensor) -> LinearOperator:
        r"""
        Returns a LinearOperator of the external potential in the grid.

        .. math::
            \mathbf{V}_{ij} = \int b_i(\mathbf{r}) V(\mathbf{r}) b_j(\mathbf{r})\ d\mathbf{r}

        External potential energy that a particle experiences in a discretized
        space or grid. In quantum mechanics or computational physics, when
        solving for the behavior of particles, an external potential is often
        introduced to represent the influence of external forces.

        Parameters
        ----------
        vext: torch.Tensor
            External potential in the grid. Shape: (`*BR`, ngrid)

        Returns
        -------
        LinearOperator
            LinearOperator of the external potential in the grid. Shape: (`*BRH`, nao, nao)

        """
        if not self.is_ao_set:
            raise RuntimeError(
                "Please call `setup_grid(grid, xc)` to call this function")
        mat = torch.einsum("...r,rb,rc->...bc", vext, self.basis_dvolume,
                           self.basis)  # (*BR, nao, nao)
        mat = self._orthozer.convert2(mat)
        mat = (
            mat + mat.transpose(-2, -1)
        ) * 0.5  # ensure the symmetricity and reduce numerical instability
        return LinearOperator.m(mat, is_hermitian=True)

    def get_vxc(
        self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
    ) -> Union[LinearOperator, SpinParam[LinearOperator]]:
        """
        Returns a LinearOperator for the exchange-correlation potential

        The exchange-correlation potential combines two effects:

        1. Exchange potential: Arises from the antisymmetry of the electron
        wave function. It quantifies the tendency of electrons to avoid each
        other due to their indistinguishability.

        2. Correlation potential: Accounts for the electron-electron
        correlation effects that arise from the repulsion between electrons.

        TODO: check if what we need for Meta-GGA involving kinetics and for
        exact-exchange

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        Union[LinearOperator, SpinParam[LinearOperator]]
            LinearOperator for the exchange-correlation potential. Shape: (`*BDH`, nao, nao)

        """
        assert self.xc is not None, "Please call .setup_grid with the xc object"

        densinfo = SpinParam.apply_fcn(lambda dm_: self._dm2densinfo(dm_),
                                       dm)  # value: (*BD, nr)
        potinfo = self.xc.get_vxc(densinfo)  # value: (*BD, nr)
        vxc_linop = SpinParam.apply_fcn(
            lambda potinfo_: self._get_vxc_from_potinfo(potinfo_), potinfo)
        return vxc_linop

    # interface to dm
    def ao_orb2dm(self, orb: torch.Tensor,
                  orb_weight: torch.Tensor) -> torch.Tensor:
        """Convert the atomic orbital to the density matrix.

        Parameters
        ----------
        orb: torch.Tensor
            Atomic orbital. Shape: (*BO, nao, norb)
        orb_weight: torch.Tensor
            Orbital weight. Shape: (*BW, norb)

        Returns
        -------
        torch.Tensor
            Density matrix. Shape: (*BOWH, nao, nao)

        """

        orb_w = orb * orb_weight.unsqueeze(-2)  # (*BOW, nao, norb)
        return torch.matmul(orb, orb_w.transpose(-2, -1))  # (*BOW, nao, nao)

    def aodm2dens(self, dm: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """Get the density value in the Cartesian coordinate.

        Parameters
        ----------
        dm: torch.Tensor
            Density matrix. Shape: (`*BD`, nao, nao)
        xyz: torch.Tensor
            Cartesian coordinate. Shape: (`*BR`, ndim)

        Returns
        -------
        torch.Tensor
            Density value in the Cartesian coordinate. Shape: (`*BRD`)

        """

        nao = dm.shape[-1]
        xyzshape = xyz.shape
        # basis: (nao, *BR)
        basis = eval_gto(self.libcint_wrapper,
                         xyz.reshape(-1, xyzshape[-1])).reshape(
                             (nao, *xyzshape[:-1]))
        basis = torch.movedim(basis, 0, -1)  # (*BR, nao)

        # torch.einsum("...ij,...i,...j->...", dm, basis, basis)
        dens = torch.matmul(dm, basis.unsqueeze(-1))  # (*BRD, nao, 1)
        dens = torch.matmul(basis.unsqueeze(-2),
                            dens).squeeze(-1).squeeze(-1)  # (*BRD)
        return dens

    # energy of the Hamiltonian
    def get_e_hcore(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Get the energy from the one-electron Hamiltonian. The input is total
        density matrix.

        Parameters
        ----------
        dm: torch.Tensor
            Total Density matrix.

        Returns
        -------
        torch.Tensor
            Energy from the one-electron Hamiltonian.

        """
        return torch.einsum("...ij,...ji->...", self.kinnucl_mat, dm)

    def get_e_elrep(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Get the energy from the electron repulsion. The input is total density
        matrix.

        Parameters
        ----------
        dm: torch.Tensor
            Total Density matrix.

        Returns
        -------
        torch.Tensor
            Energy from the one-electron Hamiltonian.

        """
        elrep_mat = self.get_elrep(dm).fullmatrix()
        return 0.5 * torch.einsum("...ij,...ji->...", elrep_mat, dm)

    def get_e_exchange(
            self, dm: Union[torch.Tensor,
                            SpinParam[torch.Tensor]]) -> torch.Tensor:
        """Get the energy from the exact exchange.

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix.

        Returns
        -------
        torch.Tensor
            Energy from the exact exchange.

        """
        exc_mat = self.get_exchange(dm)
        ene = SpinParam.apply_fcn(
            lambda exc_mat, dm: 0.5 * torch.einsum("...ij,...ji->...",
                                                   exc_mat.fullmatrix(), dm),
            exc_mat, dm)
        enetot = SpinParam.sum(ene)
        return enetot

    def get_e_xc(
            self, dm: Union[torch.Tensor,
                            SpinParam[torch.Tensor]]) -> torch.Tensor:
        """Returns the exchange-correlation energy using the xc object given in
        ``.setup_grid()``

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        torch.Tensor
            Exchange-correlation energy.

        """
        assert self.xc is not None, "Please call .setup_grid with the xc object"

        # obtain the energy density per unit volume
        densinfo = SpinParam.apply_fcn(lambda dm_: self._dm2densinfo(dm_),
                                       dm)  # (spin) value: (*BD, nr)
        edens = self.xc.get_edensityxc(densinfo)  # (*BD, nr)

        return torch.sum(self.grid.get_dvolume() * edens, dim=-1)

    # free parameters for variational method
    def ao_orb_params2dm(
        self,
        ao_orb_params: torch.Tensor,
        ao_orb_coeffs: torch.Tensor,
        orb_weight: torch.Tensor,
        with_penalty: Optional[float] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert the atomic orbital free parameters (parametrized in such a way
        so it is not bounded) to the density matrix.

        Parameters
        ----------
        ao_orb_params: torch.Tensor
            The tensor that parametrized atomic orbital in an unbounded space.
        ao_orb_coeffs: torch.Tensor
            The tensor that helps ``ao_orb_params`` in describing the orbital.
            The difference with ``ao_orb_params`` is that ``ao_orb_coeffs`` is
            not differentiable and not to be optimized in variational method.
        orb_weight: torch.Tensor
            The orbital weights.
        with_penalty: float or None
            If a float, it returns a tuple of tensors where the first element is
            ``dm``, and the second element is the penalty multiplied by the
            penalty weights. The penalty is to compensate the overparameterization
            of ``ao_orb_params``, stabilizing the Hessian for gradient calculation.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The density matrix from the orbital parameters and (if ``with_penalty``)
            the penalty of the overparameterization of ``ao_orb_params``.

        Notes
        -----
        * The penalty should be 0 if ``ao_orb_params`` is from ``dm2ao_orb_params``.
        * The density matrix should be recoverable when put through ``dm2ao_orb_params``
          and ``ao_orb_params2dm``.

        """
        ao_orbq, _ = torch.linalg.qr(ao_orb_params)  # (*BD, nao, norb)
        ao_orb = ao_orbq
        dm = self.ao_orb2dm(ao_orb, orb_weight)
        if with_penalty is None:
            return dm
        else:
            # QR decomposition's solution is not unique in a way that every column
            # can be multiplied by -1 and it still a solution
            # So, to remove the non-uniqueness, we will make the sign of the sum
            # positive.
            s1 = torch.sign(ao_orbq.sum(dim=-2, keepdim=True))  # (*BD, 1, norb)
            s2 = torch.sign(ao_orb_params.sum(dim=-2, keepdim=True))
            penalty = torch.mean(
                (ao_orbq * s1 - ao_orb_params * s2)**2) * with_penalty
            return dm, penalty

    def dm2ao_orb_params(
            self, dm: torch.Tensor, norb: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert from the density matrix to the orbital parameters.
        The map is not one-to-one, but instead one-to-many where there
        might be more than one orbital parameters to describe the same
        density matrix. For restricted systems, only one of the ``dm``
        (``dm.u`` or ``dm.d``) is sufficient.

        Parameters
        ----------
        dm: torch.Tensor
            The density matrix.
        norb: int
            The number of orbitals for the system.

        Returns
        -------
        tuple of 2 torch.Tensor
            The atomic orbital parameters for the first returned value and the
            atomic orbital coefficients for the second value.

        Note
        ----
        This assumes that the orbital weights always decreasing in order
        """
        mdmm = dm
        w, orbq = torch.linalg.eigh(mdmm)
        # w is ordered increasingly, so we take the last parts
        orbq_params = orbq[..., -norb:]  # (nao, norb)
        return torch.flip(orbq_params, dims=(-1,))

    # misc
    def _dm2densinfo(self, dm: torch.Tensor) -> ValGrad:
        """Gets Density Fitting Info from Density Matrix.

        Parameters
        ----------
        dm: torch.Tensor
            Density Matrix

        Returns
        -------
        ValGrad

        """
        # dm: (*BD, nao, nao), Hermitian
        # family: 1 for LDA, 2 for GGA, 3 for MGGA
        # self.basis: (ngrid, nao)
        # self.grad_basis: (ndim, ngrid, nao)

        ngrid = self.basis.shape[-2]
        batchshape = dm.shape[:-2]

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
        maxnumel = 16 * 1024**2 // get_dtype_memsize(self.basis)
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

    def _get_vxc_from_potinfo(self, potinfo: ValGrad) -> LinearOperator:
        """obtain the vxc operator from the potential information

        Parameters
        ----------
        potinfo: ValGrad
            potential information as ValGrad.
            potinfo.value (*BD, nr)
            potinfo.grad (*BD, ndim, nr)
            potinfo.lapl (*BD, nr)

        Returns
        -------
        LinearOperator
            vxc LinearOperator"""

        # prepare the fock matrix component from vxc
        nao = self.basis.shape[-1]
        mat = torch.zeros((*potinfo.value.shape[:-1], nao, nao),
                          dtype=self.dtype,
                          device=self.device)

        # Split the r-dimension into several parts, it is usually faster than
        # evaluating all at once
        maxnumel = 16 * 1024**2 // get_dtype_memsize(self.basis)
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
        vxc_linop = LinearOperator.m(mat, is_hermitian=True)
        return vxc_linop

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Return the paramnames

        Parameters
        ----------
        methodname: str
            Name of the method.
        prefix: str (default "")
            Prefix of the paramnames.

        Returns
        -------
        List[str]
            Paramnames.

        """
        if methodname == "get_kinnucl":
            return [prefix + "kinnucl_mat"]
        elif methodname == "get_nuclattr":
            return [prefix + "nucl_mat"]
        elif methodname == "get_overlap":
            return [prefix + "olp_mat"]
        elif methodname == "get_elrep":
            if self._df is None:
                return [prefix + "el_mat"]
            else:
                return self._df.getparamnames("get_elrep",
                                              prefix=prefix + "_df.")
        elif methodname == "get_exchange":
            return [prefix + "el_mat"]
        elif methodname == "ao_orb2dm":
            return []
        elif methodname == "ao_orb_params2dm":
            return self.getparamnames("ao_orb2dm", prefix=prefix)
        elif methodname == "get_e_hcore":
            return [prefix + "kinnucl_mat"]
        elif methodname == "get_e_elrep":
            return self.getparamnames("get_elrep", prefix=prefix)
        elif methodname == "get_e_exchange":
            return self.getparamnames("get_exchange", prefix=prefix)
        elif methodname == "get_e_xc":
            assert self.xc is not None
            return self.getparamnames("_dm2densinfo", prefix=prefix) + \
                self.xc.getparamnames("get_edensityxc", prefix=prefix + "xc.") + \
                self.grid.getparamnames("get_dvolume", prefix=prefix + "grid.")
        elif methodname == "get_vext":
            return [prefix + "basis_dvolume", prefix + "basis"] + \
                self._orthozer.getparamnames("convert2", prefix=prefix + "_orthozer.")
        elif methodname == "get_grad_vext":
            return [prefix + "basis_dvolume", prefix + "grad_basis"]
        elif methodname == "get_lapl_kin_vext":
            return [
                prefix + "dvolume", prefix + "basis", prefix + "grad_basis",
                prefix + "lapl_basis"
            ]
        elif methodname == "get_vxc":
            assert self.xc is not None
            return self.getparamnames("_dm2densinfo", prefix=prefix) + \
                self.getparamnames("_get_vxc_from_potinfo", prefix=prefix) + \
                self.xc.getparamnames("get_vxc", prefix=prefix + "xc.")
        elif methodname == "_dm2densinfo":
            params = [prefix + "basis"] + \
                self._orthozer.getparamnames("unconvert_dm", prefix=prefix + "_orthozer.")
            if self.xcfamily == 2 or self.xcfamily == 4:
                params += [prefix + "grad_basis"]
            if self.xcfamily == 4:
                params += [prefix + "lapl_basis"]
            return params
        elif methodname == "_get_vxc_from_potinfo":
            params = [prefix + "basis", prefix + "basis_dvolume"] + \
                self._orthozer.getparamnames("convert2", prefix=prefix + "_orthozer.")
            if self.xcfamily in [2, 4]:
                params += [prefix + "grad_basis"]
            if self.xcfamily == 4:
                params += [prefix + "lapl_basis", prefix + "dvolume"]
            return params
        else:
            raise KeyError("getparamnames has no %s method" % methodname)
