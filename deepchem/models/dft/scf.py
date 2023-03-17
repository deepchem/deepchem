from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Union, List, Optional, Tuple
import warnings
try:
    import torch
    import xitorch as xt
    from dqc.qccalc.ks import KS
    from dqc.utils.datastruct import SpinParam
    from deepchem.feat.dftdata import DFTEntry, DFTSystem
    from deepchem.utils.dftutils import BaseKSCalc, DQCKSCalc
    from deepchem.models.dft.nnxc import BaseNNXC, HybridXC
except ModuleNotFoundError:
    pass
from collections.abc import Callable


class Base_SCF(torch.nn.Module):
    """
    Object containing trainable parameters and the interface to the NN models.
    """

    @abstractmethod
    def get_xc(self) -> BaseNNXC:
        """
        Returns the xc model in the evaluator.
        """
        pass

    @abstractmethod
    def run(self, system: System, entry_type: str) -> BaseKSCalc:
        """
        Run the Quantum Chemistry calculation of the given system and return
        the post-run QCCalc object
        """
        pass


class XCDNN_SCF(Base_SCF):
    """
    Kohn-Sham model where the XC functional is replaced by a neural network.
    The XCDNNEvaluator is used for performing self-consistent iterations.
    Prameters
    ---------
    entries:Dict
        The entries are used to prepare buffer. The buffers are used to store
    density matrices across training epochs.
    """

    def __init__(self,
                 xc: Union[BaseNNXC, HybridXC],
                 weights: Dict[str, float],
                 always_attach: bool = False):
        super().__init__()
        self.xc = xc
        self.weights = weights
        self.always_attach = always_attach

    def get_xc(self) -> HybridXC:
        return self.xc

    def run(self, system: DFTSystem, entry_type: str) -> BaseKSCalc:
        """Run self-consistent iterations
        """
        mol = system.get_dqc_mol()
        qc = KS(mol, xc=self.xc).run()
        return DQCKSCalc(qc)
