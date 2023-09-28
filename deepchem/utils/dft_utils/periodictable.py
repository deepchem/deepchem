from typing import Union
import torch
import numpy as np
from deepchem.utils.dft_utils.datastruct import ZType

periodic_table_atomz = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
}

atom_masses = {  # isotope-averaged atom masses in a.m.u.
    # from https://www.angelo.edu/faculty/kboudrea/periodic/structure_mass.htm
    1: 1.00797,
    2: 4.00260,
    3: 6.941,
    4: 9.01218,
    5: 10.81,
    6: 12.011,
    7: 14.0067,
    8: 15.9994,
    9: 18.998403,
    10: 20.179,
    11: 22.98977,
    12: 24.305,
    13: 26.98154,
    14: 28.0855,
    15: 30.97376,
    16: 32.06,
    17: 35.453,
    18: 39.948,
    19: 39.0983,
    20: 40.08,
    21: 44.9559,
    22: 47.90,
    23: 50.9415,
    24: 51.996,
    25: 54.9380,
    26: 55.847,
    27: 58.9332,
    28: 58.70,
    29: 63.546,
    30: 65.38,
    31: 69.72,
    32: 72.59,
    33: 74.9216,
    34: 78.96,
    35: 79.904,
    36: 83.80,
    37: 85.4678,
    38: 87.62,
    39: 88.9059,
    40: 91.22,
    41: 92.9064,
    42: 95.94,
    43: 98.,
    44: 101.07,
    45: 102.9055,
    46: 106.4,
    47: 107.868,
    48: 112.41,
    49: 114.82,
    50: 118.69,
    51: 121.75,
    53: 126.9045,
    52: 127.60,
    54: 131.30,
}

# JCP 41, 3199 (1964); DOI:10.1063/1.1725697
# taken from PySCF:
# https://github.com/pyscf/pyscf/blob/45582e915e91890722fcae2bc30fb04867d5c95f/pyscf/data/radii.py#L23
# I don't know why H has 0.35 while in the reference it is 0.
# They are in angstrom, so we need to convert it to Bohr
atom_bragg_radii = list(
    np.array([
        2.00,  # Ghost atom
        0.35,
        1.40,  # 1s
        1.45,
        1.05,
        0.85,
        0.70,
        0.65,
        0.60,
        0.50,
        1.50,  # 2s2p
        1.80,
        1.50,
        1.25,
        1.10,
        1.00,
        1.00,
        1.00,
        1.80,  # 3s3p
        2.20,
        1.80,  # 4s
        1.60,
        1.40,
        1.35,
        1.40,
        1.40,
        1.40,
        1.35,
        1.35,
        1.35,
        1.35,  # 3d
        1.30,
        1.25,
        1.15,
        1.15,
        1.15,
        1.90,  # 4p
        2.35,
        2.00,  # 5s
        1.80,
        1.55,
        1.45,
        1.45,
        1.35,
        1.30,
        1.35,
        1.40,
        1.60,
        1.55,  # 4d
        1.55,
        1.45,
        1.45,
        1.40,
        1.40,
        2.10,  # 5p
        2.60,
        2.15,  # 6s
        1.95,
        1.85,
        1.85,
        1.85,
        1.85,
        1.85,
        1.85,  # La, Ce-Eu
        1.80,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,  # Gd, Tb-Lu
        1.55,
        1.45,
        1.35,
        1.35,
        1.30,
        1.35,
        1.35,
        1.35,
        1.50,  # 5d
        1.90,
        1.80,
        1.60,
        1.90,
        1.45,
        2.10,  # 6p
        1.80,
        2.15,  # 7s
        1.95,
        1.80,
        1.80,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75
    ]) / 0.52917721092)

atom_expected_radii = [
    1.0,  # Ghost atom
    1.0,  # 1
    0.927272,  # 2
    3.873661,  # 3
    2.849396,  # 4
    2.204757,  # 5
    1.714495,  # 6
    1.409631,  # 7
    1.232198,  # 8
    1.084786,  # 9
    0.965273,  # 10
    4.208762,  # 11
    3.252938,  # 12
    3.433889,  # 13
    2.752216,  # 14
    2.322712,  # 15
    2.060717,  # 16
    1.842024,  # 17
    1.662954,  # 18
    5.243652,  # 19
    4.218469,  # 20
    3.959716,  # 21
    3.778855,  # 22
    3.626288,  # 23
    3.675012,  # 24
    3.381917,  # 25
    3.258487,  # 26
    3.153572,  # 27
    3.059109,  # 28
    3.330979,  # 29
    2.897648,  # 30
    3.424103,  # 31
    2.866859,  # 32
    2.512233,  # 33
    2.299617,  # 34
    2.111601,  # 35
    1.951590,  # 36
    5.631401,  # 37
    4.632850,  # 38
    4.299870,  # 39
    4.091705,  # 40
    3.985219,  # 41
    3.841740,  # 42
    3.684647,  # 43
    3.735235,  # 44
    3.702057,  # 45
    1.533028,  # 46
    3.655961,  # 47
    3.237216,  # 48
    3.777242,  # 49
    3.248093,  # 50
    2.901067,  # 51
    2.691328,  # 52
    2.501704,  # 53
    2.337950
]  # 54


def get_atomz(element: Union[str, ZType]) -> ZType:
    """Returns the atomic number for the given element

    Examples
    --------
    >>> from deepchem.utils.dft_utils.periodictable import get_atomz
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
            raise KeyError("Element Does Not Exists or Not Documented: %s" % element)
        return atom_z
    elif isinstance(element, torch.Tensor): # Just return itself.
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
    >>> from deepchem.utils.dft_utils.periodictable import get_atom_mass
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
        atomic_mass = atom_masses[atom_z]
    except KeyError:
        raise KeyError("Element Does Not Exists or Not Documented: %d" % atom_z)

    return atomic_mass * 1822.888486209


def get_period(atom_z: int) -> int:
    """get the period of the given atom z
    
    Examples
    --------
    >>> from deepchem.utils.dft_utils.periodictable import get_period
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