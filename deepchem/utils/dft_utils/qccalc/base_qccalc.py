from abc import abstractmethod
from typing import List, Union
import torch
from deepchem.utils.dft_utils import SpinParam, BaseSystem

class BaseQCCalc(object):
    """
    Quantum Chemistry calculation. This class is the interface to the users
    regarding parameters that can be calculated after the self-consistent
    iterations (or other processes).

    This object is usually a thin-wrapper of an engine class where the
    self-consistent iteration steps (and other processes) are described.
    To avoid known memory leak, the self-consistent iteration should be
    run from this object while the steps should be described in another object
    (i.e. the engine).
    Details about the leak: https://github.com/pytorch/pytorch/issues/52140
    """
    @abstractmethod
    def get_system(self) -> BaseSystem:
        """
        Returns the system in the QC calculation
        """
        pass

    @abstractmethod
    def run(self, **kwargs):
        """
        Run the calculation.
        Note that this method can be invoked several times for one object to
        try for various self-consistent options to reach convergence.
        """
        pass

    @abstractmethod
    def energy(self) -> torch.Tensor:
        """
        Obtain the energy of the system.
        """
        pass

    @abstractmethod
    def aodm(self) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Returns the density matrix in atomic orbital. For polarized case, it
        returns a SpinParam of 2 tensors representing the density matrices for
        spin-up and spin-down.
        """
        # return: (nao, nao)
        pass

    #################### all-time calculations ####################
    # (i.e. meaning it does not have to be executed to run the functions below)
    @abstractmethod
    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Calculate the energy from the given density matrix.

        Arguments
        ---------
        dm: torch.Tensor or SpinParam of torch.Tensor
            The input density matrix. It is tensor if restricted, and SpinParam
            of tensor if unrestricted.

        Returns
        -------
        torch.Tensor
            Tensor that represents the energy given the energy.
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass
