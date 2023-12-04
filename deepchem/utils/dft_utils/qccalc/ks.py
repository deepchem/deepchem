from typing import Optional, Dict, Any, List, Union, overload, Tuple
import torch
from deepchem.utils.dft_utils.api.get_xc import get_xc
from deepchem.utils.dft_utils.qccalc.hf import _HFEngine
from deepchem.utils.differentiation_utils import LinearOperator
from deepchem.utils.dft_utils import SpinParam, BaseXC, BaseSystem, SCF_QCCalc, BaseSCFEngine

__all__ = ["KS"]

class KS(SCF_QCCalc):
    """
    Performing Restricted or Unrestricted Kohn-Sham DFT calculation.

    Arguments
    ---------
    system: BaseSystem
        The system to be calculated.
    xc: str, BaseXC, or None
        The exchange-correlation potential and energy to be used. It can accept
        ``None`` as an input to represent no xc potential involved.
    restricted: bool or None
        If True, performing restricted Kohn-Sham DFT. If False, it performs
        the unrestricted Kohn-Sham DFT.
        If None, it will choose True if the system is unpolarized and False if
        it is polarized
    variational: bool
        If True, then solve the Kohn-Sham equation variationally (i.e. using
        optimization) instead of using self-consistent iteration.
        Otherwise, solve it using self-consistent iteration.
    """

    def __init__(self, system: BaseSystem, xc: Union[str, BaseXC, None],
                 restricted: Optional[bool] = None,
                 variational: bool = False):

        engine = _KSEngine(system, xc)
        super().__init__(engine, variational)

class _KSEngine(BaseSCFEngine):
    """
    Private class of Engine to be used with KS.
    This class provides the calculation of the self-consistency iteration step
    and the calculation of the post-calculation properties.

    The reason of this class' existence is the leak in PyTorch:
    https://github.com/pytorch/pytorch/issues/52140
    which can be solved by making a different class than the class where the
    self-consistent iteration is performed.
    """
    def __init__(self, system: BaseSystem, xc: Union[str, BaseXC, None],
                 restricted: Optional[bool] = None):

        # get the xc object
        if isinstance(xc, str):
            self.xc: Optional[BaseXC] = get_xc(xc)
        elif isinstance(xc, BaseXC):
            self.xc = xc
        else:
            self.xc = xc

        # system = self.hf_engine.get_system()
        self._system = system

        # build and setup basis and grid
        self.hamilton = system.get_hamiltonian()
        if self.xc is not None or system.requires_grid():
            system.setup_grid()
            self.hamilton.setup_grid(system.get_grid(), self.xc)

        # get the HF engine and build the hamiltonian
        # no need to rebuild the grid because it has been constructed
        self.hf_engine = _HFEngine(system, restricted=restricted, build_grid_if_necessary=False)
        self._polarized = self.hf_engine.polarized

        # get the orbital info
        self.orb_weight = system.get_orbweight(polarized=self._polarized)  # (norb,)
        self.norb = SpinParam.apply_fcn(lambda orb_weight: int(orb_weight.shape[-1]),
                                        self.orb_weight)

        # set up the vext linear operator
        self.knvext_linop = self.hamilton.get_kinnucl()  # kinetic, nuclear, and external potential

    def get_system(self) -> BaseSystem:
        return self._system

    @property
    def shape(self):
        # returns the shape of the density matrix
        return self.knvext_linop.shape

    @property
    def dtype(self):
        # returns the dtype of the density matrix
        return self.knvext_linop.dtype

    @property
    def device(self):
        # returns the device of the density matrix
        return self.knvext_linop.device

    @property
    def polarized(self):
        return self._polarized

    def dm2scp(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        # convert from density matrix to a self-consistent parameter (scp)
        if isinstance(dm, torch.Tensor):  # unpolarized
            # scp is the fock matrix
            return self.__dm2fock(dm).fullmatrix()
        else:  # polarized
            # scp is the concatenated fock matrix
            fock = self.__dm2fock(dm)
            mat_u = fock.u.fullmatrix().unsqueeze(0)
            mat_d = fock.d.fullmatrix().unsqueeze(0)
            return torch.cat((mat_u, mat_d), dim=0)

    def scp2dm(self, scp: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # convert the self-consistent parameter (scp) to the density matrix
        return self.hf_engine.scp2dm(scp)

    def scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        # self-consistent iteration step from a self-consistent parameter (scp)
        # to an scp
        dm = self.scp2dm(scp)
        return self.dm2scp(dm)

    def aoparams2ene(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                     with_penalty: Optional[float] = None) -> torch.Tensor:
        # calculate the energy from the atomic orbital params
        dm, penalty = self.aoparams2dm(aoparams, aocoeffs, with_penalty)
        ene = self.dm2energy(dm)
        return (ene + penalty) if penalty is not None else ene

    def aoparams2dm(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                    with_penalty: Optional[float] = None) -> \
            Tuple[Union[torch.Tensor, SpinParam[torch.Tensor]], Optional[torch.Tensor]]:
        # calculate the density matrix and the penalty factor
        return self.hf_engine.aoparams2dm(aoparams, aocoeffs, with_penalty)

    def pack_aoparams(self, aoparams: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        # pack the aoparams from tensor or SpinParam into a single tensor
        return self.hf_engine.pack_aoparams(aoparams)

    def unpack_aoparams(self, aoparams: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # unpack the single tensor aoparams to SpinParam or a tensor
        return self.hf_engine.unpack_aoparams(aoparams)

    def set_eigen_options(self, eigen_options: Dict[str, Any]) -> None:
        # set the eigendecomposition (diagonalization) option
        self.hf_engine.set_eigen_options(eigen_options)

    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        # calculate the energy given the density matrix
        dmtot = SpinParam.sum(dm)
        e_core = self.hamilton.get_e_hcore(dmtot)
        e_elrep = self.hamilton.get_e_elrep(dmtot)
        if self.xc is not None:
            e_xc: Union[torch.Tensor, float] = self.hamilton.get_e_xc(dm)
        else:
            e_xc = 0.0
        return e_core + e_elrep + e_xc + self._system.get_nuclei_energy()

    @overload
    def __dm2fock(self, dm: torch.Tensor) -> LinearOperator:
        ...

    @overload
    def __dm2fock(self, dm: SpinParam[torch.Tensor]) -> SpinParam[LinearOperator]:
        ...

    def __dm2fock(self, dm):
        elrep = self.hamilton.get_elrep(SpinParam.sum(dm))  # (..., nao, nao)
        core_coul = self.knvext_linop + elrep

        if self.xc is not None:
            vxc = self.hamilton.get_vxc(dm)  # spin param or tensor (..., nao, nao)
            return SpinParam.apply_fcn(lambda vxc_: vxc_ + core_coul, vxc)
        else:
            if isinstance(dm, SpinParam):
                return SpinParam(u=core_coul, d=core_coul)
            else:
                return core_coul

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "scp2scp":
            return self.getparamnames("scp2dm", prefix=prefix) + \
                self.getparamnames("dm2scp", prefix=prefix)
        elif methodname == "scp2dm":
            return self.hf_engine.getparamnames("scp2dm", prefix=prefix + "hf_engine.")
        elif methodname == "dm2scp":
            return self.getparamnames("__dm2fock", prefix=prefix)
        elif methodname == "aoparams2ene":
            return self.getparamnames("aoparams2dm", prefix=prefix) + \
                self.getparamnames("dm2energy", prefix=prefix)
        elif methodname in ["aoparams2dm", "pack_aoparams", "unpack_aoparams"]:
            return self.hf_engine.getparamnames(methodname, prefix=prefix + "hf_engine.")
        elif methodname == "dm2energy":
            hprefix = prefix + "hamilton."
            sprefix = prefix + "_system."

            if self.xc is not None:
                e_xc_params = self.hamilton.getparamnames("get_e_xc", prefix=hprefix)
            else:
                e_xc_params = []

            return self.hamilton.getparamnames("get_e_hcore", prefix=hprefix) + \
                self.hamilton.getparamnames("get_e_elrep", prefix=hprefix) + \
                e_xc_params + \
                self._system.getparamnames("get_nuclei_energy", prefix=sprefix)
        elif methodname == "__dm2fock":
            hprefix = prefix + "hamilton."

            if self.xc is not None:
                vxc_params = self.hamilton.getparamnames("get_vxc", prefix=hprefix)
            else:
                vxc_params = []

            return self.hamilton.getparamnames("get_elrep", prefix=hprefix) + \
                vxc_params + \
                self.knvext_linop._getparamnames(prefix=prefix + "knvext_linop.")
        else:
            raise KeyError("Method %s has no paramnames set" % methodname)
        return []  # TODO: to complete
