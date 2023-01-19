from abc import abstractmethod
import torch
from dqc.utils.datastruct import SpinParam
from dqc.qccalc.base_qccalc import BaseQCCalc
import hashlibs

class KSCalc(object):
    """
    Interface to DQC's KS calculation.
    """

    def __init__(self, qc: BaseQCCalc):
        self.qc = qc

    def energy(self) -> torch.Tensor:
        # returns the total energy
        return self.qc.energy()

    def aodmtot(self) -> torch.Tensor:
        # returns the total density matrix
        dm = self.qc.aodm()
        if isinstance(dm, SpinParam):
            dmtot = dm.u + dm.d
        else:
            dmtot = dm
        return dmtot

    def dens(self, rgrid: torch.Tensor) -> torch.Tensor:
        # returns the total density profile in the given grid
        dmtot = self.aodmtot()
        return self.qc.get_system().get_hamiltonian().aodm2dens(dmtot, rgrid)

    def force(self) -> torch.Tensor:
        """returns the force for each atom
        The force on each atom is gradient of energy with respect to atom position.
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
    # encode the string into hashed format
    return str(hashlib.blake2s(str.encode(s)).hexdigest())
