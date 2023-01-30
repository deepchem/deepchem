"""
This file contains deprecated utilities to work with autodock vina.
"""
from deepchem.utils.docking_utils import write_vina_conf, load_docked_ligands, prepare_inputs

import warnings
import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}. Please use the corresponding function in deepchem.utils.docking_utils."
            .format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


# These functions have moved to deepchem.utils_docking_utils
write_vina_conf = deprecated(write_vina_conf)
load_docked_ligands = deprecated(load_docked_ligands)
prepare_inputs = deprecated(prepare_inputs)
