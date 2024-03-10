from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import xitorch as xt
import xitorch.linalg
import xitorch.optimize
from dqc.system.base_system import BaseSystem
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.utils.datastruct import SpinParam
from dqc.utils.config import config
from dqc.utils.misc import set_default_option

class SCF_QCCalc(BaseQCCalc):
    """
    Performing Restricted or Unrestricted self-consistent field iteration
    (e.g. Hartree-Fock or Density Functional Theory)

    Arguments
    ---------
    engine: BaseSCFEngine
        The SCF engine
    variational: bool
        If True, then use optimization of the free orbital parameters to find
        the minimum energy.
        Otherwise, use self-consistent iterations.
    """

    def __init__(self, engine: BaseSCFEngine, variational: bool = False):
        self._engine = engine
        self._polarized = engine.polarized
        self._shape = self._engine.shape
        self.dtype = self._engine.dtype
        self.device = self._engine.device
        self._has_run = False
        self._variational = variational

    def get_system(self) -> BaseSystem:
        return self._engine.get_system()

    def run(self, dm0: Optional[Union[str, torch.Tensor, SpinParam[torch.Tensor]]] = "1e",  # type: ignore
            eigen_options: Optional[Dict[str, Any]] = None,
            fwd_options: Optional[Dict[str, Any]] = None,
            bck_options: Optional[Dict[str, Any]] = None) -> BaseQCCalc:

        # get default options
        if not self._variational:
            fwd_defopt = {
                "method": "broyden1",
                "alpha": -0.5,
                "maxiter": 50,
                "verbose": config.VERBOSE > 0,
            }
        else:
            fwd_defopt = {
                "method": "gd",
                "step": 1e-2,
                "maxiter": 5000,
                "f_rtol": 1e-10,
                "x_rtol": 1e-10,
                "verbose": config.VERBOSE > 0,
            }
        bck_defopt = {
            # NOTE: it seems like in most cases the jacobian matrix is posdef
            # if it is not the case, we can just remove the line below
            "posdef": True,
        }

        # setup the default options
        if eigen_options is None:
            eigen_options = {
                "method": "exacteig"
            }
        if fwd_options is None:
            fwd_options = {}
        if bck_options is None:
            bck_options = {}
        fwd_options = set_default_option(fwd_defopt, fwd_options)
        bck_options = set_default_option(bck_defopt, bck_options)

        # save the eigen_options for use in diagonalization
        self._engine.set_eigen_options(eigen_options)

        # set up the initial self-consistent param guess
        if dm0 is None:
            dm = self._get_zero_dm()
        elif isinstance(dm0, str):
            if dm0 == "1e":  # initial density based on 1-electron Hamiltonian
                dm = self._get_zero_dm()
                scp0 = self._engine.dm2scp(dm)
                dm = self._engine.scp2dm(scp0)
            else:
                raise RuntimeError("Unknown dm0: %s" % dm0)
        else:
            dm = SpinParam.apply_fcn(lambda dm0: dm0.detach(), dm0)

        # making it spin param for polarized and tensor for nonpolarized
        if isinstance(dm, torch.Tensor) and self._polarized:
            dm_u = dm * 0.5
            dm_d = dm * 0.5
            dm = SpinParam(u=dm_u, d=dm_d)
        elif isinstance(dm, SpinParam) and not self._polarized:
            dm = dm.u + dm.d

        if not self._variational:
            scp0 = self._engine.dm2scp(dm)

            # do the self-consistent iteration
            scp = xitorch.optimize.equilibrium(
                fcn=self._engine.scp2scp,
                y0=scp0,
                bck_options={**bck_options},
                **fwd_options)

            # post-process parameters
            self._dm = self._engine.scp2dm(scp)
        else:
            system = self.get_system()
            h = system.get_hamiltonian()
            orb_weights = system.get_orbweight(polarized=self._polarized)
            norb = SpinParam.apply_fcn(lambda orb_weights: len(orb_weights), orb_weights)

            def dm2params(dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> \
                    Tuple[torch.Tensor, torch.Tensor]:
                pc = SpinParam.apply_fcn(
                     lambda dm, norb: h.dm2ao_orb_params(SpinParam.sum(dm), norb=norb),
                     dm, norb)
                p = SpinParam.apply_fcn(lambda pc: pc[0], pc)
                c = SpinParam.apply_fcn(lambda pc: pc[1], pc)
                params = self._engine.pack_aoparams(p)
                coeffs = self._engine.pack_aoparams(c)
                return params, coeffs

            def params2dm(params: torch.Tensor, coeffs: torch.Tensor) \
                    -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
                p: Union[torch.Tensor, SpinParam[torch.Tensor]] = self._engine.unpack_aoparams(params)
                c: Union[torch.Tensor, SpinParam[torch.Tensor]] = self._engine.unpack_aoparams(coeffs)

                dm = SpinParam.apply_fcn(
                    lambda p, c, orb_weights: h.ao_orb_params2dm(p, c, orb_weights, with_penalty=None),
                    p, c, orb_weights)
                return dm

            params0, coeffs0 = dm2params(dm)
            params0 = params0.detach()
            coeffs0 = coeffs0.detach()
            min_params0: torch.Tensor = xitorch.optimize.minimize(
                fcn=self._engine.aoparams2ene,
                # random noise to add the chance of it gets to the minimum, not
                # a saddle point
                y0=params0 + torch.randn_like(params0) * 0.03 / params0.numel(),
                params=(coeffs0, None,),  # coeffs & with_penalty
                bck_options={**bck_options},
                **fwd_options).detach()

            if torch.is_grad_enabled():
                # If the gradient is required, then put it through the minimization
                # one more time with penalty on the parameters.
                # The penalty is to keep the Hamiltonian invertible, stabilizing
                # inverse.
                # Without the penalty, the Hamiltonian could have 0 eigenvalues
                # because of the overparameterization of the aoparams.
                min_dm = params2dm(min_params0, coeffs0)
                params0, coeffs0 = dm2params(min_dm)
                min_params0 = xitorch.optimize.minimize(
                    fcn=self._engine.aoparams2ene,
                    y0=params0,
                    params=(coeffs0, 1e-1,),  # coeffs & with_penalty
                    bck_options={**bck_options},
                    method="gd",
                    step=0,
                    maxiter=0)

            self._dm = params2dm(min_params0, coeffs0)

        self._has_run = True
        return self

    def energy(self) -> torch.Tensor:
        # returns the total energy of the system
        assert self._has_run
        return self._engine.dm2energy(self._dm)

    def aodm(self) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # returns the density matrix in the atomic-orbital basis
        assert self._has_run
        return self._dm

    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]):
        # calculate the energy given the density matrix
        assert (isinstance(dm, torch.Tensor) and not self._polarized) or \
            (isinstance(dm, SpinParam) and self._polarized)
        return self._engine.dm2energy(dm)

    def _get_zero_dm(self) -> Union[SpinParam[torch.Tensor], torch.Tensor]:
        # get the initial dm that are all zeros
        if not self._polarized:
            return torch.zeros(self._shape, dtype=self.dtype,
                               device=self.device)
        else:
            dm0_u = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            dm0_d = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            return SpinParam(u=dm0_u, d=dm0_d)

class BaseSCFEngine(xt.EditableModule):
    @abstractproperty
    def polarized(self) -> bool:
        """
        Returns if the system is polarized or not
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Returns the shape of the density matrix in this engine.
        """
        pass

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        pass

    @abstractproperty
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        pass

    @abstractmethod
    def get_system(self) -> BaseSystem:
        """
        Returns the system involved in the engine.
        """
        pass

    @abstractmethod
    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Calculate the energy from the given density matrix.
        """
        pass

    @abstractmethod
    def dm2scp(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Convert the density matrix into the self-consistent parameter (scp).
        Self-consistent parameter is defined as the parameter that is put into
        the equilibrium function, i.e. y in `y = f(y, x)`.
        """
        pass

    @abstractmethod
    def scp2dm(self, scp: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Calculate the density matrix from the given self-consistent parameter (scp).
        """
        pass

    @abstractmethod
    def scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        """
        Calculate the next self-consistent parameter (scp) for the next iteration
        from the previous scp.
        """
        pass

    @abstractmethod
    def aoparams2ene(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                     with_penalty: Optional[float] = None) -> torch.Tensor:
        """
        Calculate the energy from the given atomic orbital parameters and coefficients.
        """
        pass

    @abstractmethod
    def aoparams2dm(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                    with_penalty: Optional[float] = None) -> \
            Tuple[Union[torch.Tensor, SpinParam[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Calculate the density matrix and the penalty from the given atomic
        orbital parameters and coefficients.
        """
        pass

    @abstractmethod
    def pack_aoparams(self, aoparams: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Pack the ao params into a single tensor.
        """
        pass

    @abstractmethod
    def unpack_aoparams(self, aoparams: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Unpack the ao params into a tensor or SpinParam of tensor.
        """
        pass

    @abstractmethod
    def set_eigen_options(self, eigen_options: Dict[str, Any]) -> None:
        """
        Set the options for the diagonalization (i.e. eigendecomposition).
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        List all the names of parameters used in the given method.
        """
        pass
