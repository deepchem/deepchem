from __future__ import annotations
import os
from abc import abstractmethod, abstractproperty
from typing import List, Dict, Optional, Union
import numpy as np
import torch
try:
    import dqc
    from dqc.system.mol import Mol
    from dqc.system.base_system import BaseSystem
    from dqc.grid.base_grid import BaseGrid
    from deepchem.utils.dftutils import KSCalc
except ModuleNotFoundError:
    raise ModuleNotFoundError("This utility requires dqc")

import yaml
from yaml.loader import SafeLoader


class DFTSystem(dict):
    """
    """

    created_systems: Dict[str, DFTSystem] = {}

    @classmethod
    def create(cls, system: Dict) -> DFTSystem:
        # create the system if it has not been created
        # otherwise, return the previously created system

        system_str = str(system)
        if system_str not in cls.created_systems:
            cls.created_systems[system_str] = DFTSystem(system)
        return cls.created_systems[system_str]

    def __init__(self, system: Dict):
        super().__init__(system)

        # caches
        self._caches = {}

    def get_dqc_system(self, pos_reqgrad: bool = False) -> BaseSystem:
        # convert the system dictionary to DQC system

        systype = self["type"]
        if systype == "mol":
            atomzs, atomposs = dqc.parse_moldesc(self["kwargs"]["moldesc"])
            if pos_reqgrad:
                atomposs.requires_grad_()
            mol = Mol(**self["kwargs"])
            return mol
        else:
            raise RuntimeError("Unknown system type: %s" % systype)

    def get_cache(self, s: str):
        return self._caches.get(s, None)

    def set_cache(self, s: str, obj) -> None:
        self._caches[s] = obj


class DFTEntry(dict):
    """
    Interface to the entry of the dataset.
    Entry class should not be initialized directly, but created through
    ``Entry.create``
    """

    created_entries: Dict[str, DFTEntry] = {}

    @classmethod
    def create(cls,
               entry_dct: Union[Dict, DFTEntry],
               device: torch.device,
               dtype: torch.dtype = torch.double) -> DFTEntry:
        if isinstance(entry_dct, DFTEntry):
            # TODO: should we add dtype and device checks here?
            return entry_dct

        s = str(entry_dct)
        if s not in cls.created_entries:
            tpe = entry_dct["type"]
            kwargs = {
                "entry_dct": entry_dct,
                "dtype": dtype,
                "device": device,
            }
            obj = {
                "ae": EntryAE,
                "ie": EntryIE,
                "dm": EntryDM,
                "dens": EntryDens,
                "force": EntryForce,
            }[tpe](**kwargs)
            cls.created_entries[s] = obj
        return cls.created_entries[s]

    def __init__(self,
                 entry_dct: Dict,
                 device: torch.device,
                 dtype: torch.dtype = torch.double):
        super().__init__(entry_dct)
        self._systems = [DFTSystem.create(p) for p in entry_dct["systems"]]
        self._dtype = dtype
        self._device = device

        self._trueval_is_set = False
        self._trueval = torch.tensor(0.0).to(device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def get_systems(self) -> List[DFTSystem]:
        """
        Returns the list of systems in the entry
        """
        return self._systems

    @abstractproperty
    def entry_type(self) -> str:
        """
        Returning the type of the entry of the dataset
        """
        pass

    def get_true_val(self) -> torch.Tensor:
        if not self._trueval_is_set:
            self._trueval = self._get_true_val()
            self._trueval_is_set = True
        return self._trueval

    @abstractmethod
    def _get_true_val(self) -> torch.Tensor:
        """
        Get the true value of the entry.
        """
        pass

    @abstractmethod
    def get_val(self, qcs: List[KSCalc]) -> torch.Tensor:
        """
        Calculate the value of the entry given post-run QC objects.
        """
        pass


class EntryDM(DFTEntry):
    """Entry for Density Matrix (DM)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.get_systems()) == 1, "dm entry can only have 1 system"

    @property
    def entry_type(self) -> str:
        return "dm"

    def _get_true_val(self) -> torch.Tensor:
        # get the density matrix from PySCF's CCSD calculation
        dm = np.load(self["true_val"])
        true_val = torch.from_numpy(dm)
        return torch.as_tensor(true_val, device=self.device)

    def get_val(self, qcs: List[KSCalc]) -> torch.Tensor:
        return qcs[0].aodmtot()


class EntryDens(DFTEntry):
    """Entry for density profile (dens), compared with CCSD calculation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.get_systems()) == 1, "dens entry can only have 1 system"
        self._grid: Optional[BaseGrid] = None

    @property
    def entry_type(self) -> str:
        return "dens"

    def _get_true_val(self) -> torch.Tensor:
        # get the density profile from PySCF's CCSD calculation

        # get the density matrix from the PySCF calculation
        system = self.get_systems()[0]
        dens = np.load(self["trueval"])
        true_val = torch.from_numpy(dens)
        return torch.as_tensor(true_val, device=self.device)

    def get_val(self, qcs: List[KSCalc]) -> torch.Tensor:
        qc = qcs[0]

        # get the integration grid infos
        grid = self._get_integration_grid()
        rgrid = grid.get_rgrid()

        # get the density profile
        return qc.dens(rgrid)

    def get_integration_grid(self) -> BaseGrid:
        if self._grid is None:
            system = self.get_systems()[0]

            # reduce the level of grid
            orig_grid_level: Optional[int] = None
            if "grid" in system:
                orig_grid_level = system["grid"]

            # get the dqc grid
            system["grid"] = "sg2"
            dqc_mol = system.get_dqc_system()
            dqc_mol.setup_grid()
            grid = dqc_mol.get_grid()
            assert grid.coord_type == "cart"

            # restore the grid level
            if orig_grid_level is not None:
                system["grid"] = orig_grid_level
            else:
                system.pop("grid")

            self._grid = grid

        return self._grid


class EntryForce(DFTEntry):
    """Entry for force at the experimental equilibrium position"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(
            self.get_systems()) == 1, "force entry can only have 1 system"

    @property
    def entry_type(self) -> str:
        return "force"

    def _get_true_val(self) -> torch.Tensor:
        # get the density matrix from PySCF's CCSD calculation
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    def get_val(self, qcs: List[KSCalc]) -> torch.Tensor:
        return qcs[0].force()


class EntryIE(DFTEntry):
    """Entry for Ionization Energy (IE)"""

    @property
    def entry_type(self) -> str:
        return "ie"

    def _get_true_val(self) -> torch.Tensor:
        return torch.as_tensor(self["true_val"],
                               dtype=self.dtype,
                               device=self.device)

    def get_val(self, qcs: List[KSCalc]) -> torch.Tensor:
        glob = {"systems": qcs, "energy": self.energy}
        return eval(self["cmd"], glob)


class EntryAE(EntryIE):
    """Entry for Atomization Energy (AE)"""

    @property
    def entry_type(self) -> str:
        return "ae"


def load_entries(entry_path, device):
    entries = []
    with open(entry_path) as f:
        data_mol = yaml.load(f, Loader=SafeLoader)
    for i in range(0, len(data_mol)):
        entry = DFTEntry.create(data_mol[i], device=device)
        entries.append(entry)
    return entries
