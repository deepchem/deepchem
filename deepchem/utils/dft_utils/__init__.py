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
    from deepchem.utils.dft_utils.data.datastruct import AtomCGTOBasis
    from deepchem.utils.dft_utils.data.datastruct import BasisInpType

    from deepchem.utils.dft_utils.hamilton.orbparams import BaseOrbParams
    from deepchem.utils.dft_utils.hamilton.orbparams import QROrbParams
    from deepchem.utils.dft_utils.hamilton.orbparams import MatExpOrbParams
    from deepchem.utils.dft_utils.hamilton.intor.lcintwrap import LibcintWrapper
    from deepchem.utils.dft_utils.hamilton.intor.lcintwrap import SubsetLibcintWrapper
    from deepchem.utils.dft_utils.hamilton.intor.molintor import int1e
    from deepchem.utils.dft_utils.hamilton.intor.molintor import int2c2e
    from deepchem.utils.dft_utils.hamilton.intor.molintor import int3c2e
    from deepchem.utils.dft_utils.hamilton.intor.molintor import int2e
    from deepchem.utils.dft_utils.hamilton.intor.molintor import overlap
    from deepchem.utils.dft_utils.hamilton.intor.molintor import kinetic
    from deepchem.utils.dft_utils.hamilton.intor.molintor import nuclattr
    from deepchem.utils.dft_utils.hamilton.intor.molintor import elrep
    from deepchem.utils.dft_utils.hamilton.intor.molintor import coul2c
    from deepchem.utils.dft_utils.hamilton.intor.molintor import coul3c

    from deepchem.utils.dft_utils.api.parser import parse_moldesc
    from deepchem.utils.dft_utils.api.loadbasis import loadbasis

    from deepchem.utils.dft_utils.grid.base_grid import BaseGrid

    from deepchem.utils.dft_utils.grid.radial_grid import RadialGrid
    from deepchem.utils.dft_utils.grid.radial_grid import get_xw_integration
    from deepchem.utils.dft_utils.grid.radial_grid import SlicedRadialGrid
    from deepchem.utils.dft_utils.grid.radial_grid import BaseGridTransform
    from deepchem.utils.dft_utils.grid.radial_grid import DE2Transformation
    from deepchem.utils.dft_utils.grid.radial_grid import LogM3Transformation
    from deepchem.utils.dft_utils.grid.radial_grid import TreutlerM4Transformation
    from deepchem.utils.dft_utils.grid.radial_grid import get_grid_transform

    from deepchem.utils.dft_utils.xc.base_xc import BaseXC
    from deepchem.utils.dft_utils.xc.base_xc import AddBaseXC
    from deepchem.utils.dft_utils.xc.base_xc import MulBaseXC
    from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcLDALibXCPol
    from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcLDALibXCUnpol
    from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcGGALibXCUnpol
    from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcGGALibXCPol
    from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcMGGALibXCUnpol
    from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcMGGALibXCPol

    from deepchem.utils.dft_utils.xc.libxc import LibXCLDA
    from deepchem.utils.dft_utils.xc.libxc import LibXCGGA
    from deepchem.utils.dft_utils.xc.libxc import LibXCMGGA

    from deepchem.utils.dft_utils.api.getxc import get_libxc
    from deepchem.utils.dft_utils.api.getxc import get_xc

    from deepchem.utils.dft_utils.df.base_df import BaseDF

    from deepchem.utils.dft_utils.hamilton.base_hamilton import BaseHamilton

    from deepchem.utils.dft_utils.system.base_system import BaseSystem

    from deepchem.utils.dft_utils.qccalc.base_qccalc import BaseQCCalc
    from deepchem.utils.dft_utils.qccalc.scf_qccalc import SCF_QCCalc
    from deepchem.utils.dft_utils.qccalc.scf_qccalc import BaseSCFEngine
    from deepchem.utils.dft_utils.qccalc.hf import HF
    from deepchem.utils.dft_utils.qccalc.hf import HFEngine
    from deepchem.utils.dft_utils.qccalc.ks import KS
    from deepchem.utils.dft_utils.qccalc.ks import KSEngine
except ModuleNotFoundError as e:
    logger_.warning(
        f'Skipped loading some Pytorch utilities, missing a dependency. {e}')
