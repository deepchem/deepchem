"""
Density Functional Theory utilities.
"""
# flake8: noqa
import logging

logger_ = logging.getLogger(__name__)

from deepchem.utils.dft_utils.config import config

try:
    from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice

    from deepchem.utils.dft_utils.api.parser import parse_moldesc

    from deepchem.utils.dft_utils.xc.base_xc import BaseXC

    from deepchem.utils.dft_utils.datastruct import ZType
    from deepchem.utils.dft_utils.datastruct import SpinParam
    from deepchem.utils.dft_utils.datastruct import ValGrad
except ModuleNotFoundError as e:
    logger_.warning(
        f'Skipped loading some Pytorch utilities, missing a dependency. {e}')
