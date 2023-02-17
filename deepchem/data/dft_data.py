"""
Density Functional Theory Data
Derived from: https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/entry.py
"""
from __future__ import annotations
from abc import abstractmethod, abstractproperty
from typing import List, Dict, Optional, Union
import numpy as np
try:
    import dqc
    from dqc.system.mol import Mol
    from dqc.system.base_system import BaseSystem
    from dqc.grid.base_grid import BaseGrid
    from deepchem.utils.dftutils import KSCalc
except ModuleNotFoundError:
    raise ModuleNotFoundError("This data class requires dqc")


class DFTSystem(dict):
    """
    The DFTSystem class creates and returns the various systems in an entry object as dictionaries.

    Examples
    --------
    >>> from deepchem.data.dft_data import DFTSystem
    >>> entry.get_systems() #for any entry object

    Returns
    -------
    List of dictionaries for all the different atoms/ions/molecules in an entry object.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.

    https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/system/base_system.py
    """

    created_systems: Dict[str, DFTSystem] = {}
    @classmethod
    def create(cls, system: Dict) -> DFTSystem:
        """
        Returns
        -------
        Creates and returns the system if it has not been created
        manually during training step. Otherwise, return the previously created system.
        """
        system_str = str(system)
        if system_str not in cls.created_systems:
            cls.created_systems[system_str] = DFTSystem(system)
        return cls.created_systems[system_str]

    def __init__(self, system: Dict):
        super().__init__(system)

    def get_dqc_mol(self, pos_reqgrad: bool = False) -> BaseSystem:
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


class DFTEntry(dict):
    """
    Handles creating and initialising DFTEntry objects from the dataset. This object contains information    about the various systems in the datapoint (atoms, molecules and ions) along with the ground truth
    values.
    Notes
    -----
    Entry class should not be initialized directly, but created through
    ``Entry.create``

    Example
    -------
    >>> from deepchem.data.dft_data import DFTEntry
    >>> data_mol = {
        'name':
            'Density matrix of HF',
        'type':
            'dm',
        'cmd':
            'dm(systems[0])',
        'true_val':
            'output.npy',
        'systems': [{
            'type': 'mol',
            'kwargs': {
                'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
                'basis': '6-311++G(3df,3pd)'
            }
        }]
    }
    >>>dm_entry_for_HF = DFTEntry.create(data_mol)
    """

    created_entries: Dict[str, DFTEntry] = {}

    @classmethod
    def create(cls, entry_dct: Union[Dict, DFTEntry]) -> DFTEntry:
        """
        This method is used to initialise the DFTEntry class. The entry objects are created
        based on their entry type.

        Parameters
        ----------
            entry_dct: Dict

        Returns
        -------
        created_entries: Dict[str, DFTEntry]
        A dictionary containing multiple entry objects

        """
        if isinstance(entry_dct, DFTEntry):
            return entry_dct

        s = str(entry_dct)
        if s not in cls.created_entries:
            tpe = entry_dct["type"]
            if tpe == "ae":
                obj = _EntryAE(entry_dct)
            elif tpe == "ie":
                obj = _EntryIE(entry_dct)
            elif tpe == "dm":
                obj = _EntryDM(entry_dct)
            elif tpe == "dens":
                obj = _EntryDens(entry_dct)
            cls.created_entries[s] = obj
        return cls.created_entries[s]

    def __init__(self, entry_dct: Dict):
        super().__init__(entry_dct)
        self._systems = [DFTSystem.create(p) for p in entry_dct["systems"]]

        self._trueval_is_set = False
        self._trueval = np.ndarray
        """
        Parameters
        ----------
        systems:  List[DFTSystem]

        Returns
        -------
        List of systems in the entry
        """

    def get_systems(self) -> List[DFTSystem]:
        """
        Returns
        -------
        List of systems in the entry
        """
        return self._systems

    @abstractproperty
    def entry_type(self) -> str:
        """
        Returns
        -------
        The type of entry ;
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


class _EntryDM(DFTEntry):
    """Entry for Density Matrix (DM)"""

    def __init__(self, entry_dct):
        super().__init__(entry_dct)
        assert len(self.get_systems()) == 1, "dm entry can only have 1 system"

    @property
    def entry_type(self) -> str:
        return "dm"

    def _get_true_val(self) -> np.ndarray:
        # get the density matrix from PySCF's CCSD calculation
        dm = np.load(self["true_val"])
        return dm

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        return (qcs.aodmtot()).numpy()


class _EntryDens(DFTEntry):
    """Entry for density profile (dens), compared with CCSD calculation"""

    def __init__(self, entry_dct):
        super().__init__(entry_dct)
        assert len(self.get_systems()) == 1
        self._grid: Optional[BaseGrid] = None

    @property
    def entry_type(self) -> str:
        return "dens"

    def _get_true_val(self) -> np.ndarray:
        dens = np.load(self["trueval"])
        return dens

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        qc = qcs[0]

        grid = self._get_integration_grid()
        rgrid = grid.get_rgrid()

        return (qc.dens(rgrid)).numpy()

    def get_integration_grid(self) -> BaseGrid:
        """
        This method is used to calculate the integration grid required for a system
        in order to calculate it's density profile using Differentiable DFT.

        Returns
        -------
            grid: BaseGrid

        References
        ----------
        https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/grid/base_grid.py
        """
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


class _EntryIE(DFTEntry):
    """Entry for Ionization Energy (IE)"""

    @property
    def entry_type(self) -> str:
        return "ie"

    def _get_true_val(self) -> np.ndarray:
        return (self["true_val"])

    def get_val(self, qcs: List[KSCalc]) -> np.ndarray:
        glob = {"systems": qcs, "energy": self.energy}
        return eval(self["cmd"], glob)


class _EntryAE(_EntryIE):
    """Entry for Atomization Energy (AE)"""

    @property
    def entry_type(self) -> str:
        return "ae"
