"""
Density Functional Theory Data
Derived from: https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/entry.py
"""
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
    The DFTSystem class creates and returns the various systems in an entry object as dictionaries. 

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.
    
    https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/system/base_system.py
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
        """
        This method converts the system dictionary to a DQC system and returns it. 
        Parameters
        ----------
        pos_reqgrad: bool 
            decides if the atomic position require gradient calculation. 
        """
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
    Handles creating and initialising DFTEntry objects from the dataset. This object contains information    about the various systems in the datapoint (atoms, molecules and ions) along with the ground truth
    values.    
    Note: Entry class should not be initialized directly, but created through
    ``Entry.create``
    """

    created_entries: Dict[str, DFTEntry] = {}

    @classmethod
    def create(cls,
               entry_dct: Union[Dict, DFTEntry]) -> DFTEntry:
        if isinstance(entry_dct, DFTEntry):
            return entry_dct

        s = str(entry_dct)
        if s not in cls.created_entries:
            tpe = entry_dct["type"]
            kwargs = {
                "entry_dct": entry_dct,
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
                 entry_dct: Dict):
        super().__init__(entry_dct)
        self._systems = [DFTSystem.create(p) for p in entry_dct["systems"]]

        self._trueval_is_set = False
        self._trueval = np.ndarray
        """
        Parameters
        ----------
        systems:  List[DFTSystem]
            Returns the list of systems in the entry
        """


    def get_systems(self) -> List[DFTSystem]:
        """
        Returns the list of systems in the entry
        """
        return self._systems

    @abstractproperty
    def entry_type(self) -> str:
        """
        Returning the type of the entry of the dataset; 
        1) Atomic Ionization Potential (IP/IE)
        2) Atomization Energy (AE)
        3) Density Profile (DENS)
        4) Density Matrix (DM)
        """
        pass

    def get_true_val(self) -> np.ndarray:
        if not self._trueval_is_set:
            self._trueval = self._get_true_val()
            self._trueval_is_set = True
        return self._trueval

    @abstractmethod
    def _get_true_val(self) -> np.ndarray:
        """
        Get the true value of the DFTEntry.  
        For the AE and IP entry types, the experimental values are collected from the NIST CCCBDB/ASD 
        databases. 
        The true values of density profiles are calculated using PYSCF-CCSD calculations. This method            simply loads the value, no calculation is performed.  
        """
        pass

    @abstractmethod
    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        """
        Return the energy value of the entry, using a DQC-DFT calculation, where the XC has been 
        replaced by the trained neural network. This method does not carry out any calculations, it is 
        an interface to the KSCalc utility. 
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

    def _get_true_val(self) -> np.ndarray:
        # get the density matrix from PySCF's CCSD calculation
        dm = np.load(self["true_val"])
        return dm

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        return (qcs[0].aodmtot()).numpy()


class EntryDens(DFTEntry):
    """Entry for density profile (dens), compared with CCSD calculation"""

    def __init__(self, *args, **kwargs):

        self._grid: Optional[BaseGrid] = None

    @property
    def entry_type(self) -> str:
        return "dens"

    def _get_true_val(self) -> np.ndarray:
        # get the density profile from PySCF's CCSD calculation

        system = self.get_systems()[0]
        dens = np.load(self["trueval"])
        return dens 

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        qc = qcs[0]

        # get the integration grid infos
        grid = self._get_integration_grid()
        rgrid = grid.get_rgrid()

        # get the density profile
        return (qc.dens(rgrid)).numpy()

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

    def _get_true_val(self) -> np.ndarray:
        # get the density matrix from PySCF's CCSD calculation
        return np.array(0.0)

    def get_val(self, qcs: List[KSCalc]) -> torch.Tensor:
        return (qcs[0].force()).numpy()


class EntryIE(DFTEntry):
    """Entry for Ionization Energy (IE)"""

    @property
    def entry_type(self) -> str:
        return "ie"

    def _get_true_val(self) -> np.ndarray:
        return (self["true_val"])

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        glob = {"systems": qcs, "energy": self.energy}
        return eval(self["cmd"], glob)


class EntryAE(EntryIE):
    """Entry for Atomization Energy (AE)"""

    @property
    def entry_type(self) -> str:
        return "ae"


def load_entries(entry_path):
    """
    The method loads the yaml dataset and returns a list of entries, containing DFTEntry objects. 
    """
    entries = []
    with open(entry_path) as f:
        data_mol = yaml.load(f, Loader=SafeLoader)
    for i in range(0, len(data_mol)):
        entry = DFTEntry.create(data_mol[i])
        entries.append(entry)
    return entries
