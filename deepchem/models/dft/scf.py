from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Union, List, Optional, Tuple
import warnings
import torch
import xitorch as xt
from dqc.qccalc.ks import KS
from dqc.utils.datastruct import SpinParam
from deepchem.feat.dft_data import DFTEntry, DFTSystem
from deepchem.utils.dftutils import KSCalc, hashstr
from deepchem.models.dft.nnxc import BaseNNXC, HybridXC
from dqc.utils.datastruct import SpinParam

from collections.abc import Callable


class BaseSCF(torch.nn.Module):
    """
    """

    @abstractmethod
    def get_xc(self) -> BaseNNXC:
        """
        Returns the xc model in the evaluator.
        """
        pass

    @abstractmethod
    def run(self, system: DFTSystem) -> KSCalc:
        """
        Run the Quantum Chemistry calculation of the given system and return
        the post-run QCCalc object
        """
        pass


class XCNNSCF(BaseSCF):
    """
    Kohn-Sham model where the XC functional is replaced by a neural network.
    The XCDNNEvaluator is used for performing self-consistent iterations.
    Prameters
    ---------
    entries:Dict
    """

    def __init__(self,
                 xc: Union[BaseNNXC, HybridXC],
                 entry: DFTEntry,
                 always_attach: bool = False):
        super().__init__()
        self.xc = xc
        self.always_attach = always_attach

    def get_xc(self) -> HybridXC:
        return self.xc

    def run(self, system: DFTSystem) -> KSCalc:
        """Run self-consistent iterations
        """
        dm0, dmname = self._get_dm0(system)
        mol = system.get_dqc_mol()
        qc = KS(mol, xc=self.xc).run(dm0=dm0, bck_options={"max_niter": 50})
        return KSCalc(qc)

    def _dm0_name(self, obj) -> str:
        return "dm0_" + hashstr(str(obj))


    def _get_dm0(self, system: DFTSystem):
        dm_name = self._dm0_name(system)
        dm0: torch.Tensor = getattr(self, dm_name, None)
        dm_exists = dm0 is not None
        dm_written = dm_exists and torch.any(dm0 != 0.0)
        if not dm_written:
            dm0_res: Union[None, torch.Tensor, SpinParam[torch.Tensor]] = None
        elif system.get_dqc_mol().spin != 0:
            dm0_res = SpinParam(u=dm0[0].detach(), d=dm0[1].detach())
        else:
            dm0_res = dm0

        return dm0_res, (dm_name if dm_exists else None)
