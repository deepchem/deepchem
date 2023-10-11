from __future__ import annotations
from abc import abstractmethod, abstractproperty
import torch
import xitorch as xt
from typing import List, Union, Optional, Tuple
from dqc.hamilton.base_hamilton import BaseHamilton
from deepchem.models.dft.grid.base_grid import BaseGrid
from deepchem.utils.dft_utils.datastruct import SpinParam, ZType, BasisInpType

class BaseSystem(xt.EditableModule):
    """
    System is a class describing the environment before doing the quantum
    chemistry calculation.
    """
    @abstractmethod
    def densityfit(self, method: Optional[str] = None,
                   auxbasis: Optional[BasisInpType] = None) -> BaseSystem:
        """
        Indicate that the system's Hamiltonian will use density fitting.
        """
        pass

    @abstractmethod
    def get_hamiltonian(self) -> BaseHamilton:
        """
        Returns the Hamiltonian object for the system
        """
        pass

    @abstractmethod
    def set_cache(self, fname: str, paramnames: Optional[List[str]] = None) -> BaseSystem:
        """
        Set up the cache to read/write some parameters from the given files.
        If paramnames is not given, then read/write all cache-able parameters
        specified by each class.
        Returns self
        """
        pass

    @abstractmethod
    def get_orbweight(self, polarized: bool = False) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Returns the atomic orbital weights. If polarized == False, then it
        returns the total orbital weights. Otherwise, it returns a tuple of
        orbital weights for spin-up and spin-down.
        """
        # returns: (*BS, norb)
        pass

    @abstractmethod
    def get_nuclei_energy(self) -> torch.Tensor:
        """
        Returns the nuclei-nuclei repulsion energy.
        """
        pass

    @abstractmethod
    def setup_grid(self) -> None:
        """
        Construct the integration grid for the system
        """
        pass

    @abstractmethod
    def get_grid(self) -> BaseGrid:
        """
        Returns the grid of the system
        """
        pass

    @abstractmethod
    def requires_grid(self) -> bool:
        """
        True if the system needs the grid to be constructed. Otherwise, returns
        False
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass

    @abstractmethod
    def make_copy(self, **kwargs) -> BaseSystem:
        """
        Returns a copy of the system identical to the orginal except for new
        parameters set in the kwargs.
        """
        pass

    ####################### system properties #######################
    @abstractproperty
    def atompos(self) -> torch.Tensor:
        """
        Returns the atom positions as a tensor with shape ``(natoms, ndim)``
        """
        pass

    @abstractproperty
    def atomzs(self) -> torch.Tensor:
        """
        Returns the tensor containing the atomic number with shape ``(natoms,)``
        """
        pass

    @abstractproperty
    def atommasses(self) -> torch.Tensor:
        """
        Returns the tensor containing atomic mass with shape ``(natoms)`` in atomic unit
        """
        pass

    @abstractproperty
    def spin(self) -> ZType:
        """
        Returns the total spin of the system.
        """
        pass

    @abstractproperty
    def charge(self) -> ZType:
        """
        Returns the charge of the system.
        """
        pass

    @abstractproperty
    def numel(self) -> ZType:
        """
        Returns the total number of the electrons in the system.
        """
        pass

    @abstractproperty
    def efield(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Returns the external electric field of the system, or None if there is
        no electric field.
        """
        pass