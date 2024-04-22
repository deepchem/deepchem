"""Part of this code adopted from https://github.com/diffqc/dqc"""
from typing import Optional, Dict, Any, Tuple, List, Union
import torch
from deepchem.utils.dft_utils import BaseSystem, SCF_QCCalc, BaseSCFEngine, SpinParam
from deepchem.utils.differentiation_utils import LinearOperator, lsymeig


class HF(SCF_QCCalc):
    """
    Performing Restricted or Unrestricted Kohn-Sham DFT calculation.

    Parameters
    ----------
    system: BaseSystem
        The system to be calculated.
    restricted: bool or None
        If True, performing restricted Kohn-Sham DFT. If False, it performs
        the unrestricted Kohn-Sham DFT.
        If None, it will choose True if the system is unpolarized and False if
        it is polarized
    variational: bool
        If True, then use optimization of the free orbital parameters to find
        the minimum energy.
        Otherwise, use self-consistent iterations.

    """

    def __init__(self,
                 system: BaseSystem,
                 restricted: Optional[bool] = None,
                 variational: bool = False):

        engine = HFEngine(system, restricted)
        super().__init__(engine, variational)


class HFEngine(BaseSCFEngine):
    """
    Engine to be used with Hartree Fock.
    This class provides the calculation of the self-consistency iteration step
    and the calculation of the post-calculation properties.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import HFEngine, BaseSystem, BaseHamilton, SpinParam, BaseGrid
    >>> from deepchem.utils.differentiation_utils import LinearOperator
    >>> from typing import List, Optional
    >>> class MyLinOp(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(MyLinOp, self).__init__(shape)
    ...         self.param = torch.rand(shape)
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix + "param"]
    ...     def _mv(self, x):
    ...         return torch.matmul(self.param, x)
    ...     def _rmv(self, x):
    ...         return torch.matmul(self.param.transpose(-2,-1).conj(), x)
    ...     def _mm(self, x):
    ...         return torch.matmul(self.param, x)
    ...     def _rmm(self, x):
    ...         return torch.matmul(self.param.transpose(-2,-1).conj(), x)
    ...     def _fullmatrix(self):
    ...         return self.param
    >>> class MyHamilton(BaseHamilton):
    ...     def __init__(self):
    ...         self._nao = 2
    ...         self._kpts = torch.tensor([[0.0, 0.0, 0.0]])
    ...         self._df = None
    ...     @property
    ...     def nao(self):
    ...         return self._nao
    ...     @property
    ...     def kpts(self):
    ...         return self._kpts
    ...     @property
    ...     def df(self):
    ...         return self._df
    ...     def build(self):
    ...         return self
    ...     def get_nuclattr(self):
    ...         return torch.ones((1, 1, self.nao, self.nao))
    ...     def get_e_elrep(self, dmtot):
    ...         return 2 * dmtot
    ...     def get_e_exchange(self, dm):
    ...         if isinstance(dm, SpinParam):
    ...             return SpinParam.sum(dm)
    ...         else:
    ...             return 2 * dm
    ...     def get_e_hcore(self, dm):
    ...         return 4 * dm
    ...     def get_elrep(self, dmtot):
    ...         return MyLinOp((self.nao + 1, self.nao + 1))
    ...     def get_exchange(self, dm):
    ...         return MyLinOp((self.nao + 1, self.nao + 1))
    ...     def get_kinnucl(self):
    ...         linop = MyLinOp((self.nao + 1, self.nao + 1))
    ...         return linop
    ...     def ao_orb2dm(self, orb: torch.Tensor,
    ...                   orb_weight: torch.Tensor) -> torch.Tensor:
    ...         return orb * orb_weight
    ...     def ao_orb_params2dm(
    ...         self,
    ...         ao_orb_params: torch.Tensor,
    ...         ao_orb_coeffs: torch.Tensor,
    ...         orb_weight: torch.Tensor,
    ...         with_penalty: Optional[float] = None) -> List[torch.Tensor]:
    ...         return [ao_orb_params * orb_weight, ao_orb_coeffs * orb_weight]
    >>> ham = MyHamilton()
    >>> class MySystem(BaseSystem):
    ...     def __init__(self):
    ...         self.hamiltonian = ham
    ...         self.grid = BaseGrid()
    ...     def get_hamiltonian(self):
    ...         return self.hamiltonian
    ...     def get_grid(self):
    ...         return self.grid
    ...     def requires_grid(self):
    ...         return True
    ...     def get_orbweight(self, polarized: bool = False) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
    ...         return SpinParam(torch.tensor([2.0]), torch.tensor([2.0]))
    ...     def get_nuclei_energy(self):
    ...         return torch.tensor(10.0)
    >>> system = MySystem()
    >>> engine = HFEngine(system, False)
    >>> engine.set_eigen_options(eigen_options={"method": "exacteig"})
    >>> engine.dm2energy(torch.tensor([2])).shape
    torch.Size([1])
    >>> engine.dm2scp(torch.tensor([2])).shape
    torch.Size([3, 3])
    >>> engine.scp2dm(torch.rand((2, 2, 2))).u.shape
    torch.Size([2, 1])

    """

    def __init__(self,
                 system: BaseSystem,
                 restricted: Optional[bool] = None,
                 build_grid_if_necessary: bool = False):

        # decide if this is restricted or not
        if restricted is None:
            self._polarized = bool(system.spin != 0)
        else:
            self._polarized = not restricted

        # construct the grid if the system requires it
        if build_grid_if_necessary and system.requires_grid():
            system.setup_grid()
            system.get_hamiltonian().setup_grid(system.get_grid())

        # build the basis
        self._hamilton = system.get_hamiltonian().build()
        self._system = system

        # get the orbital info
        self._orb_weight = system.get_orbweight(
            polarized=self._polarized)  # (norb,)
        self._norb = SpinParam.apply_fcn(
            lambda orb_weight: int(orb_weight.shape[-1]), self._orb_weight)

        # set up the 1-electron linear operator
        self._core1e_linop = self._hamilton.get_kinnucl()  # kinetic and nuclear

    def get_system(self) -> BaseSystem:
        """Return the system object.

        Returns
        -------
        BaseSystem
            The system object.

        """
        return self._system

    @property
    def shape(self):
        """Shape of the density matrix

        Returns
        -------
        Tuple[int, int]
            Shape of the density matrix.

        """
        return self._core1e_linop.shape

    @property
    def dtype(self):
        """Dtype of the density matrix

        Returns
        -------
        torch.dtype
            Dtype of the density matrix.

        """
        return self._core1e_linop.dtype

    @property
    def device(self):
        """Device of the density matrix

        Returns
        -------
        torch.device
            Device of the density matrix.

        """
        return self._core1e_linop.device

    @property
    def polarized(self):
        """Returns if the calculation is polarized

        Returns
        -------
        bool
            If the calculation is polarized.

        """
        return self._polarized

    def dm2scp(
            self, dm: Union[torch.Tensor,
                            SpinParam[torch.Tensor]]) -> torch.Tensor:
        """Convert from density matrix to a self-consistent parameter (scp)

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix to be converted.

        Returns
        -------
        torch.Tensor
            Self-consistent parameter.

        """
        if isinstance(dm, torch.Tensor):  # unpolarized
            # scp is the fock matrix
            fork = self.__dm2fock(dm)
            assert isinstance(fork, LinearOperator), type(fork)
            return fork.fullmatrix()
        else:  # polarized
            # scp is the concatenated fock matrix
            fock = self.__dm2fock(dm)
            assert isinstance(fock, SpinParam), type(fock)
            mat_u = fock.u.fullmatrix().unsqueeze(0)
            mat_d = fock.d.fullmatrix().unsqueeze(0)
            return torch.cat((mat_u, mat_d), dim=0)

    def scp2dm(
            self,
            scp: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """Convert from self-consistent parameter (scp) to density matrix

        Parameters
        ----------
        scp: torch.Tensor
            Self-consistent parameter to be converted.

        Returns
        -------
        Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix.

        """
        if not self._polarized:
            fock = LinearOperator.m(_symm(scp), is_hermitian=True)
            return self.__fock2dm(fock)
        else:
            fock_u = LinearOperator.m(_symm(scp[0]), is_hermitian=True)
            fock_d = LinearOperator.m(_symm(scp[1]), is_hermitian=True)
            return self.__fock2dm(SpinParam(u=fock_u, d=fock_d))

    def scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        """Self-consistent iteration step from a self-consistent parameter (scp)
        to an scp

        Parameters
        ----------
        scp: torch.Tensor
            Self-consistent parameter to be converted.

        Returns
        -------
        torch.Tensor
            New self-consistent parameter.

        """
        dm = self.scp2dm(scp)
        return self.dm2scp(dm)

    def aoparams2ene(self,
                     aoparams: torch.Tensor,
                     aocoeffs: torch.Tensor,
                     with_penalty: Optional[float] = None) -> torch.Tensor:
        """Calculate the energy from the atomic orbital params

        Parameters
        ----------
        aoparams: torch.Tensor
            Atomic orbital parameters.
        aocoeffs: torch.Tensor
            Atomic orbital coefficients.
        with_penalty: Optional[float]
            Penalty factor to be added to the energy.

        Returns
        -------
        torch.Tensor
            Energy value.

        """
        dm, penalty = self.aoparams2dm(aoparams, aocoeffs, with_penalty)
        ene = self.dm2energy(dm)
        return (ene + penalty) if penalty is not None else ene

    def aoparams2dm(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                    with_penalty: Optional[float] = None) -> \
            Tuple[Union[torch.Tensor, SpinParam[torch.Tensor]], Optional[torch.Tensor]]:
        """Convert the aoparams to density matrix and penalty factor

        Parameters
        ----------
        aoparams: torch.Tensor
            Atomic orbital parameters.
        aocoeffs: torch.Tensor
            Atomic orbital coefficients.
        with_penalty: Optional[float]
            Penalty factor to be added to the energy.

        Returns
        -------
        Tuple[Union[torch.Tensor, SpinParam[torch.Tensor]], Optional[torch.Tensor]]
            Density matrix and the penalty factor.

        """
        aop = self.unpack_aoparams(aoparams)  # tensor or SpinParam of tensor
        aoc = self.unpack_aoparams(aocoeffs)  # tensor or SpinParam of tensor
        dm_penalty = SpinParam.apply_fcn(
            lambda aop, aoc, orb_weight: self._hamilton.ao_orb_params2dm(
                aop, aoc, orb_weight, with_penalty=with_penalty), aop, aoc,
            self._orb_weight)
        if with_penalty is not None:
            dm = SpinParam.apply_fcn(lambda dm_penalty: dm_penalty[0],
                                     dm_penalty)
            penalty: Optional[torch.Tensor] = SpinParam.sum(
                SpinParam.apply_fcn(lambda dm_penalty: dm_penalty[1],
                                    dm_penalty))
        else:
            dm = dm_penalty
            penalty = None
        return dm, penalty

    def pack_aoparams(
            self, aoparams: Union[torch.Tensor,
                                  SpinParam[torch.Tensor]]) -> torch.Tensor:
        """Check if polarized, then pack it by concatenating them in the last dimension

        Parameters
        ----------
        aoparams: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Atomic orbital parameters.

        Returns
        -------
        torch.Tensor
            Packed atomic orbital parameters.

        """
        if isinstance(aoparams, SpinParam):
            return torch.cat((aoparams.u, aoparams.d), dim=-1)
        else:
            return aoparams

    def unpack_aoparams(
            self, aoparams: torch.Tensor
    ) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """Check if polarized, then construct the SpinParam (reverting the pack_aoparams)

        Parameters
        ----------
        aoparams: torch.Tensor
            Packed atomic orbital parameters.

        Returns
        -------
        Union[torch.Tensor, SpinParam[torch.Tensor]]
            Atomic orbital parameters.

        """
        if isinstance(self._norb, SpinParam):
            return SpinParam(u=aoparams[..., :self._norb.u],
                             d=aoparams[..., self._norb.u:])
        else:
            return aoparams

    def set_eigen_options(self, eigen_options: Dict[str, Any]) -> None:
        """Set the eigendecomposition (diagonalization) option

        Parameters
        ----------
        eigen_options: Dict[str, Any]
            Options for the eigendecomposition.

        """
        self.eigen_options = eigen_options

    def dm2energy(
            self, dm: Union[torch.Tensor,
                            SpinParam[torch.Tensor]]) -> torch.Tensor:
        """Calculate the energy given the density matrix

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix.

        Returns
        -------
        torch.Tensor
            Energy.

        """
        dmtot = SpinParam.sum(dm)
        e_core = self._hamilton.get_e_hcore(dmtot)
        e_elrep = self._hamilton.get_e_elrep(dmtot)
        e_exch = self._hamilton.get_e_exchange(dm)
        return e_core + e_elrep + e_exch + self._system.get_nuclei_energy()

    def __dm2fock(
        self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
    ) -> Union[LinearOperator, SpinParam[LinearOperator]]:
        """From density matrix, returns the fock matrix

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix.

        Returns
        -------
        Union[LinearOperator, SpinParam[LinearOperator]]
            Fock matrix.

        """
        vhf = self.__dm2vhf(dm)
        fock = SpinParam.apply_fcn(lambda vhf: self._core1e_linop + vhf, vhf)
        return fock

    def __dm2vhf(
        self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
    ) -> Union[LinearOperator, SpinParam[LinearOperator]]:
        """from density matrix, returns the linear operator on electron-electron
        coulomb and exchange

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix.

        Returns
        -------
        Union[LinearOperator, SpinParam[LinearOperator]]
            Linear operator on electron-electron coulomb and exchange

        """
        elrep = self._hamilton.get_elrep(SpinParam.sum(dm))
        exch = self._hamilton.get_exchange(dm)
        vhf = SpinParam.apply_fcn(lambda exch: elrep + exch, exch)
        return vhf

    def __fock2dm(
        self, fock: Union[LinearOperator, SpinParam[LinearOperator]]
    ) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """Diagonalize the fock matrix and obtain the density matrix

        Parameters
        ----------
        fock: Union[LinearOperator, SpinParam[LinearOperator]]
            Fock matrix.

        Returns
        -------
        Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix.

        """
        eigvals, eigvecs = self.diagonalize(fock, self._norb)
        dm = SpinParam.apply_fcn(
            lambda eivecs, orb_weights: self._hamilton.ao_orb2dm(
                eivecs, orb_weights), eigvecs, self._orb_weight)
        return dm

    def diagonalize(self, fock: Union[LinearOperator, SpinParam[LinearOperator]], norb: Union[int, SpinParam[int]]) -> \
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[SpinParam[torch.Tensor], SpinParam[torch.Tensor]]]:
        """Diagonalize the fock matrix

        Parameters
        ----------
        fock: Union[LinearOperator, SpinParam[LinearOperator]]
            Fock matrix.
        norb: Union[int, SpinParam[int]]
            Number of orbitals.

        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[SpinParam[torch.Tensor], SpinParam[torch.Tensor]]]
            Eigenvalues and eigenvectors.

        """
        ovlp = self._hamilton.get_overlap()
        if isinstance(fock, SpinParam):
            assert isinstance(self._norb, SpinParam), type(self._norb)
            assert isinstance(norb, SpinParam), type(norb)
            eivals_u, eivecs_u = lsymeig(A=fock.u,
                                         neig=norb.u,
                                         M=ovlp,
                                         **self.eigen_options)
            eivals_d, eivecs_d = lsymeig(A=fock.d,
                                         neig=norb.d,
                                         M=ovlp,
                                         **self.eigen_options)
            return SpinParam(u=eivals_u, d=eivals_d), SpinParam(u=eivecs_u,
                                                                d=eivecs_d)
        else:
            assert isinstance(norb, int)
            return lsymeig(A=fock, neig=norb, M=ovlp, **self.eigen_options)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Parameter names for the given method

        Parameters
        ----------
        methodname: str
            Method name.
        prefix: str
            Prefix to be added to the parameter names.

        Returns
        -------
        List[str]
            Parameter names.

        """
        if methodname == "scp2scp":
            return self.getparamnames("scp2dm", prefix=prefix) + \
                self.getparamnames("dm2scp", prefix=prefix)
        elif methodname == "scp2dm":
            return self.getparamnames("__fock2dm", prefix=prefix)
        elif methodname == "dm2scp":
            return self.getparamnames("__dm2fock", prefix=prefix)
        elif methodname == "aoparams2ene":
            return self.getparamnames("aoparams2dm", prefix=prefix) + \
                self.getparamnames("dm2energy", prefix=prefix)
        elif methodname == "aoparams2dm":
            if isinstance(self._orb_weight, SpinParam):
                params = [prefix + "_orb_weight.u", prefix + "_orb_weight.d"]
            else:
                params = [prefix + "_orb_weight"]
            return params + \
                self._hamilton.getparamnames("ao_orb_params2dm", prefix=prefix + "_hamilton.")
        elif methodname == "pack_aoparams":
            return []
        elif methodname == "unpack_aoparams":
            return []
        elif methodname == "dm2energy":
            hprefix = prefix + "_hamilton."
            sprefix = prefix + "_system."
            return self._hamilton.getparamnames("get_e_hcore", prefix=hprefix) + \
                self._hamilton.getparamnames("get_e_elrep", prefix=hprefix) + \
                self._hamilton.getparamnames("get_e_exchange", prefix=hprefix) + \
                self._system.getparamnames("get_nuclei_energy", prefix=sprefix)
        elif methodname == "__fock2dm":
            if isinstance(self._orb_weight, SpinParam):
                params = [prefix + "_orb_weight.u", prefix + "_orb_weight.d"]
            else:
                params = [prefix + "_orb_weight"]
            return self.getparamnames("diagonalize", prefix=prefix) + \
                self._hamilton.getparamnames("ao_orb2dm", prefix=prefix + "_hamilton.") + \
                params
        elif methodname == "__dm2fock":
            return self._core1e_linop._getparamnames(prefix=prefix + "_core1e_linop.") + \
                self.getparamnames("__dm2vhf", prefix=prefix)
        elif methodname == "__dm2vhf":
            hprefix = prefix + "_hamilton."
            return self._hamilton.getparamnames("get_elrep", prefix=hprefix) + \
                self._hamilton.getparamnames("get_exchange", prefix=hprefix)
        elif methodname == "diagonalize":
            return self._hamilton.getparamnames("get_overlap",
                                                prefix=prefix + "_hamilton.")
        else:
            raise KeyError("Method %s has no paramnames set" % methodname)


def _symm(scp: torch.Tensor):
    """Forcely symmetrize the tensor

    Parameters
    ----------
    scp: torch.Tensor
        Tensor to be symmetrized.

    Returns
    -------
    torch.Tensor
        Symmetrized tensor.

    """
    return (scp + scp.transpose(-2, -1)) * 0.5
