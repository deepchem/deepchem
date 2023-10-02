"""
Density Functional Theory utilities.
"""
# flake8: noqa
import logging

logger_ = logging.getLogger(__name__)

from deepchem.utils.dft_utils.config import config

try:
    from deepchem.utils.dft_utils.datastruct import ZType
except ModuleNotFoundError as e:
    logger_.warning(
        f'Skipped loading some Pytorch utilities, missing a dependency. {e}')
