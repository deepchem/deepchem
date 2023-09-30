"""
Density Functional Theory utilities.
"""
# flake8: noqa
import logging

logger_ = logging.getLogger(__name__)

from deepchem.utils.dft_utils.config import config

try:
    from deepchem.utils.dft_utils.mem import chunkify
    from deepchem.utils.dft_utils.mem import get_memory
    from deepchem.utils.dft_utils.mem import get_dtype_memsize

    from deepchem.utils.dft_utils.periodictable import get_atomz
    from deepchem.utils.dft_utils.periodictable import get_atom_mass
    from deepchem.utils.dft_utils.periodictable import get_period

    from deepchem.utils.dft_utils.datastruct import ZType

    from deepchem.utils.dft_utils.misc import set_default_option
    from deepchem.utils.dft_utils.misc import memoize_method
    from deepchem.utils.dft_utils.misc import get_option
    from deepchem.utils.dft_utils.misc import gaussian_int
    from deepchem.utils.dft_utils.misc import logger
except ModuleNotFoundError as e:
    logger_.warning(
        f'Skipped loading some Pytorch utilities, missing a dependency. {e}')
