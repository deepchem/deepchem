"""
Density Functional Theory utilities.
"""
# flake8: noqa
import logging

logger_ = logging.getLogger(__name__)

from deepchem.utils.dft_utils.config import config

try:
    from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice

    from deepchem.utils.dft_utils.data.datastruct import ZType
    from deepchem.utils.dft_utils.data.datastruct import AtomPosType
    from deepchem.utils.dft_utils.data.datastruct import AtomZsType
    from deepchem.utils.dft_utils.data.datastruct import SpinParam
    from deepchem.utils.dft_utils.data.datastruct import ValGrad
    from deepchem.utils.dft_utils.data.datastruct import CGTOBasis
    from deepchem.utils.dft_utils.data.datastruct import BasisInpType
    from deepchem.utils.dft_utils.data.datastruct import AtomCGTOBasis
    from deepchem.utils.dft_utils.data.datastruct import is_z_float
    from deepchem.utils.dft_utils.data.datastruct import DensityFitInfo

    from deepchem.utils.dft_utils.hamilton.orbparams import BaseOrbParams
    from deepchem.utils.dft_utils.hamilton.orbparams import QROrbParams
    from deepchem.utils.dft_utils.hamilton.orbparams import MatExpOrbParams

    from deepchem.utils.dft_utils.api.parser import parse_moldesc
    from deepchem.utils.dft_utils.api.loadbasis import loadbasis

    from deepchem.utils.dft_utils.xc.base_xc import BaseXC
    from deepchem.utils.dft_utils.xc.base_xc import AddBaseXC

    from deepchem.utils.dft_utils.grid.base_grid import BaseGrid

    from deepchem.utils.dft_utils.df.base_df import BaseDF

    from deepchem.utils.dft_utils.hamilton.base_hamilton import BaseHamilton

    from deepchem.utils.dft_utils.system.base_system import BaseSystem

    from deepchem.utils.dft_utils.data.safeops import occnumber
    from deepchem.utils.dft_utils.data.safeops import get_floor_and_ceil
    from deepchem.utils.dft_utils.data.safeops import safe_cdist

    from deepchem.utils.dft_utils.hamilton.intor.lcintwrap import *
    from deepchem.utils.dft_utils.hamilton.intor.molintor import *
    from deepchem.utils.dft_utils.hamilton.intor.gtoeval import *

    from deepchem.utils.dft_utils.hamilton.orbconverter import BaseOrbConverter
    from deepchem.utils.dft_utils.hamilton.orbconverter import OrbitalOrthogonalizer
    from deepchem.utils.dft_utils.hamilton.orbconverter import IdentityOrbConverter

    from deepchem.utils.dft_utils.df.dfmol import DFMol

    from deepchem.utils.dft_utils.data.pbc import unweighted_coul_ft
    from deepchem.utils.dft_utils.data.pbc import estimate_ovlp_rcut
    from deepchem.utils.dft_utils.data.pbc import estimate_g_cutoff
    from deepchem.utils.dft_utils.data.pbc import get_gcut

    from deepchem.utils.dft_utils.hamilton.hcgto import HamiltonCGTO
except ModuleNotFoundError as e:
    logger_.warning(
        f'Skipped loading some Pytorch utilities, missing a dependency. {e}')
