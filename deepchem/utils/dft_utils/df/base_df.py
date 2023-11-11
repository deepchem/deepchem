from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import List
import torch
from deepchem.utils.differentiation_utils import EditableModule, LinearOperator


class BaseDF(EditableModule):
    """
    BaseDF represents the density fitting object used in calculating the
    electron repulsion (and xc energy?) in Hamiltonian.
    
    Density fitting in density functional theory (DFT) is a technique used to
    reduce the computational cost of evaluating electron repulsion integrals.
    In DFT, the key quantity is the electron density rather than the wave
    function, and the electron repulsion integrals involve four-electron
    interactions, making them computationally demanding.

    Density fitting exploits the fact that many-electron integrals can be
    expressed as a sum of two-electron integrals involving auxiliary basis
    functions. By approximating these auxiliary basis functions, often referred
    to as fitting functions, the computational cost can be significantly reduced.

    """

    @abstractmethod
    def build(self) -> BaseDF:
        """
        Construct the matrices required to perform the calculation and return
        self.
        """
        pass

    @abstractmethod
    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        """
        Construct the electron repulsion linear operator from the given density
        matrix using the density fitting method.
        """
        pass

    ################ properties ################
    @abstractproperty
    def j2c(self) -> torch.Tensor:
        """
        Returns the 2-centre 2-electron integrals of the auxiliary basis.
        """
        pass

    @abstractproperty
    def j3c(self) -> torch.Tensor:
        """
        Return the 3-centre 2-electron integrals of the auxiliary basis and the
        basis.
        """
        pass

    ################ properties ################
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass
