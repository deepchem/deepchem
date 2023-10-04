"""
Density Function Theory Periodic Table Utilities.
"""
from typing import Union
import json
import os
import numpy as np

import torch
from deepchem.utils.dft_utils.datastruct import ZType

with open(
        os.path.join(os.path.dirname(__file__), "assets",
                     "periodic_table.json")) as read_file:
    file = json.load(read_file)

periodic_table_atomz = file["periodic_table_atomz"]

# isotope-averaged atom masses in a.m.u.
# from https://www.angelo.edu/faculty/kboudrea/periodic/structure_mass.htm
atom_masses = file["atom_masses"]

# JCP 41, 3199 (1964); DOI:10.1063/1.1725697
# taken from PySCF:
# https://github.com/pyscf/pyscf/blob/45582e915e91890722fcae2bc30fb04867d5c95f/pyscf/data/radii.py#L23
# I don't know why H has 0.35 while in the reference it is 0.
# They are in angstrom, so we need to convert it to Bohr
atom_bragg_radii = list(np.array(file["atom_bragg_radii"]) / 0.52917721092)

atom_expected_radii = file["atom_expected_radii"]


def get_atomz(element: Union[str, ZType]) -> ZType:
    """Returns the atomic number for the given element

    Examples
    --------
    >>> from deepchem.utils.periodictable_utils import get_atomz
    >>> element_symbol = "Al" # Aluminium
    >>> get_atomz(element_symbol)
    13
    >>> get_atomz(17)
    17

    Parameters
    ----------
    element: Union[str, ZType]
        String symbol of Element or Atomic Number.
        Ex: H, He, C

    Returns
    -------
    atom_n: ZType
        Atomic Number of the given Element.

    """
    if isinstance(element, str):
        try:
            atom_z = periodic_table_atomz[element]
        except KeyError:
            raise KeyError("Element Does Not Exists or Not Documented: %s" %
                           element)
        return atom_z
    elif isinstance(element, torch.Tensor):  # Just return itself.
        try:
            assert element.numel() == 1
        except:
            raise AssertionError("Only 1 element Tensor Allowed.")
        return element
    else:  # float or int | Just return itself.
        return element


def get_atom_mass(atom_z: int) -> float:
    """Returns the Atomic mass in Atomic Mass Unit.

    Examples
    --------
    >>> from deepchem.utils.periodictable_utils import get_atom_mass
    >>> atom_number = 13
    >>> get_atom_mass(atom_number)
    49184.33860618758

    Parameters
    ----------
    atom_z: int
        Atomic Number of the Element.

    Returns
    -------
    atomic_mass: float
        Atomic Mass of the Element.

    """
    try:
        atomic_mass = atom_masses[str(atom_z)]
    except KeyError:
        raise KeyError("Element Does Not Exists or Not Documented: %d" % atom_z)

    return atomic_mass * 1822.888486209


def get_period(atom_z: int) -> int:
    """get the period of the given atom z

    Examples
    --------
    >>> from deepchem.utils.periodictable_utils import get_period
    >>> atom_number = 13
    >>> get_period(atom_number)
    3

    Parameters
    ----------
    atom_z: int
        Atomic Number of the Element.

    Returns
    -------
    period: int
        Period of the Element.

    """
    period: int = 0

    if atom_z <= 2:
        period = 1
    elif atom_z <= 10:
        period = 2
    elif atom_z <= 18:
        period = 3
    elif atom_z <= 36:
        period = 4
    elif atom_z <= 54:
        period = 5
    elif atom_z <= 86:
        period = 6
    elif atom_z <= 118:
        period = 7
    else:
        raise RuntimeError("Unimplemented atomz: %d" % atom_z)

    return period
