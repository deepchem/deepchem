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
    from deepchem.utils.dft_utils.data.datastruct import SpinParam
    from deepchem.utils.dft_utils.data.datastruct import ValGrad

    from deepchem.utils.dft_utils.xc.base_xc import BaseXC
    from deepchem.utils.dft_utils.xc.base_xc import AddBaseXC
    from deepchem.utils.dft_utils.xc.base_xc import MulBaseXC


    from deepchem.utils.dft_utils.api.parser import parse_moldesc
except ModuleNotFoundError as e:
    logger_.warning(
        f'Skipped loading some Pytorch utilities, missing a dependency. {e}')
