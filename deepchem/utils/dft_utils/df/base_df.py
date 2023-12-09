from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import List
import torch
from deepchem.utils.differentiation_utils import EditableModule, LinearOperator


class BaseDF(EditableModule):
    """
    BaseDF represents the density fitting object used in calculating the
    electron repulsion (and xc energy) in Hamiltonian.

    Density fitting in density functional theory (DFT) is a technique used to
    reduce the computational cost of evaluating electron repulsion integrals.
    In DFT, the key quantity is the electron density rather than the wave
    function, and the electron repulsion integrals involve four-electron
    interactions, making them computationally demanding.

    Density fitting exploits the fact that many-electron integrals can be
    expressed as a sum of two-electron integrals involving auxiliary basis
    functions. By approximating these auxiliary basis functions, often referred
    to as fitting functions, the computational cost can be significantly reduced.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import BaseDF
    >>> import torch
    >>> class MyDF(BaseDF):
    ...     def __init__(self):
    ...         super(MyDF, self).__init__()
    ...     def get_j2c(self):
    ...         return torch.ones((3, 3))
    ...     def get_j3c(self):
    ...         return torch.ones((3, 3, 3))
    >>> df = MyDF()
    >>> df.get_j2c()
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])

    """

    @abstractmethod
    def build(self) -> BaseDF:
        """
        Construct the matrices required to perform the calculation and return
        self.

        Returns
        -------
        BaseDF
            The constructed density fitting object.

        """
        pass

    @abstractmethod
    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        """
        Construct the electron repulsion linear operator from the given density
        matrix using the density fitting method.

        Parameters
        ----------
        dm : torch.Tensor
            The density matrix.

        Returns
        -------
        LinearOperator
            The electron repulsion linear operator.

        """
        pass

    # properties
    @abstractproperty
    def j2c(self) -> torch.Tensor:
        """Returns the 2-centre 2-electron integrals of the auxiliary basis.

        Returns
        -------
        torch.Tensor
            The 2-centre 2-electron integrals of the auxiliary basis.

        """
        pass

    @abstractproperty
    def j3c(self) -> torch.Tensor:
        """
        Return the 3-centre 2-electron integrals of the auxiliary basis and the
        basis.

        Returns
        -------
        torch.Tensor
            The 3-centre 2-electron integrals of the auxiliary basis and the
            basis.

        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        This method should list tensor names that affect the output of the
        method with name indicated in ``methodname``.

        Parameters
        ---------
        methodname: str
            The name of the method of the class.
        prefix: str (default="")
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            Sequence of name of parameters affecting the output of the method.

        """
        pass
