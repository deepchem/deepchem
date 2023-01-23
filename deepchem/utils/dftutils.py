"""
Density Functional Theory Utilities
Derived from: https://github.com/mfkasim1/xcnn/blob/main/xcdnn2/kscalc.py
"""
import torch
from dqc.utils.datastruct import SpinParam
from dqc.qccalc.base_qccalc import BaseQCCalc
import hashlib


class _KSCalc(object):
    """
    Interface to DQC's KS calculation.

    Parameters
    __________
    qc: BaseQCCalc
        object often acts as a wrapper around an engine class (from dqc.qccalc) that contains information about the self-consistent iterations.
    References
    __________
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/master/dqc/qccalc/ks.py
    """

    def __init__(self, qc: BaseQCCalc):
        self.qc = qc

    def energy(self) -> torch.Tensor:
        """
        Returns
        _______
        The total energy
        """
        return self.qc.energy()

    def aodmtot(self) -> torch.Tensor:
        """
        Returns
        _______
        The total density matrix
        """
        dm = self.qc.aodm()
        if isinstance(dm, SpinParam):
            dmtot = dm.u + dm.d
        else:
            dmtot = dm
        return dmtot

    def dens(self, rgrid: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        __________
        rgrid: torch.Tensor
            Calculate integration grid using dqc.grid.
        Returns
        _______
        The total density profile in the given grid

        Reference
        __________
        https://github.com/diffqc/dqc/blob/master/dqc/grid/base_grid.py
        """
        dmtot = self.aodmtot()
        return self.qc.get_system().get_hamiltonian().aodm2dens(dmtot, rgrid)

    def force(self) -> torch.Tensor:
        """
        Returns
        _______
        The force for each atom
        It is calculated as the gradient of energy with respect to         the atomic position.
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
    Encodes the string into hashed format

    Parameters
    ----------
    s : str
    """
    return str(hashlib.blake2s(str.encode(s)).hexdigest())
