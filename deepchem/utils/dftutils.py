"""
Density Functional Theory Utilities
Derived from: https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/kscalc.py """
import hashlib
import warnings
from abc import abstractmethod, abstractproperty
from typing import Union, List
try:
    import torch
except Exception as e:
    warnings.warn("Could not import torch. Skipping tests." + str(e))

from deepchem.utils.differentiation_utils import EditableModule
from deepchem.utils.dft_utils import BaseQCCalc, SpinParam


class KSCalc(object):
    """
    Interface to DQC's KS calculation.

    Parameters
    ----------
    qc: BaseQCCalc
        object often acts as a wrapper around an engine class (from dqc.qccalc) that contains information about the self-consistent iterations.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/master/dqc/qccalc/ks.py
    """

    def __init__(self, qc: BaseQCCalc):
        self.qc = qc

    def energy(self) -> torch.Tensor:
        """
        Returns
        -------
        The total energy of the Kohn-Sham calculation for a particular system.
        """
        return self.qc.energy()

    def aodmtot(self) -> torch.Tensor:
        """
        Both interacting and non-interacting system's total energy can be expressed in terms of the density matrix. The ground state properties of a system can be calculated by minimizing the energy w.r.t the density matrix.

        Returns
        -------
        The total density matrix in atomic orbital bases.
        """
        dm = self.qc.aodm()
        if isinstance(dm, SpinParam):
            dmtot: torch.Tensor = dm.u + dm.d
        else:
            # mypy needs explicit assertion that dm is torch.Tensor here
            assert isinstance(dm, torch.Tensor)
            dmtot = dm
        return dmtot

    def dens(self, rgrid: torch.Tensor) -> torch.Tensor:
        """
        The ground state density n(r) of a system.

        Parameters
        ----------
        rgrid: torch.Tensor
            Calculate integration grid using dqc.grid.

        Returns
        -------
        The total density profile in the given grid

        Reference
        ---------
        https://github.com/diffqc/dqc/blob/master/dqc/grid/base_grid.py
        """
        dmtot = self.aodmtot()
        return self.qc.get_system().get_hamiltonian().aodm2dens(dmtot, rgrid)

    def force(self) -> torch.Tensor:
        """
        The force on an atom is calculated as the gradient of energy with respect to the atomic position.

        Returns
        -------
        The force for each atom.
        """
        ene = self.energy()
        atompos = self.qc.get_system().atompos
        is_grad_enabled = torch.is_grad_enabled()
        f, = torch.autograd.grad(ene,
                                 atompos,
                                 create_graph=is_grad_enabled,
                                 retain_graph=True)
        return f


def hashstr(s: str) -> str:
    """
    Encodes the string into hashed format - hexadecimal digits.

    Parameters
    ----------
    s : str
    """
    return str(hashlib.blake2s(str.encode(s)).hexdigest())


class BaseGrid(EditableModule):
    """
    Interface to DQC's BaseGrid class. BaseGrid is a class that regulates the integration points over the spatial
    dimensions.
    Parameters
    ----------
    qc: BaseQCCalc
        object often acts as a wrapper around an engine class (from dqc.qccalc) that contains information about the self-consistent iterations.
    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/grid/base_grid.py
    """

    @abstractproperty
    def dtype(self) -> torch.dtype:
        pass

    @abstractproperty
    def device(self) -> torch.device:
        pass

    @abstractproperty
    def coord_type(self) -> str:
        """
        Returns the type of the coordinate returned in get_rgrid
        """
        pass

    @abstractmethod
    def get_dvolume(self) -> torch.Tensor:
        """
        Obtain the torch.tensor containing the dV elements for the integration.
        Returns
        -------
        torch.tensor (*BG, ngrid)
            The dV elements for the integration. *BG is the length of the BaseGrid.
        """
        pass

    @abstractmethod
    def get_rgrid(self) -> torch.Tensor:
        """
        Returns the grid points position in the specified coordinate in
        self.coord_type.
        Returns
        -------
        torch.tensor (*BG, ngrid, ndim)
            The grid points position. *BG is the length of the BaseGrid.
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        Return a list with the parameter names corresponding to the given method
        (methodname)
        Returns
        -------
        List[str]
            List of parameter names of methodname
        """
        pass
