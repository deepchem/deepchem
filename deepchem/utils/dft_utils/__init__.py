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

    from deepchem.utils.dft_utils.hamilton.orbparams import BaseOrbParams
    from deepchem.utils.dft_utils.hamilton.orbparams import QROrbParams
    from deepchem.utils.dft_utils.hamilton.orbparams import MatExpOrbParams

    from deepchem.utils.dft_utils.api.parser import parse_moldesc

    from deepchem.utils.dft_utils.xc.base_xc import BaseXC
except ModuleNotFoundError as e:
    logger_.warning(
        f'Skipped loading some Pytorch utilities, missing a dependency. {e}')
