"""
Tests for DFT Periodic Table Utilities.
"""
try:
    from deepchem.utils.periodic_table_utils import get_atomz, get_atom_mass, get_period
    has_torch = True
except:
    has_torch = False
import numpy as np
import pytest


@pytest.mark.torch
def test_get_atomz():
    element_symbol = "Al"  # Aluminium
    atomic_number = get_atomz(element_symbol)
    assert atomic_number == 13  # Aluminium Atom Number


def test_get_atom_mass():
    atomic_number = 41
    atomic_mass = get_atom_mass(atomic_number)
    assert np.allclose(atomic_mass, 92.9064 * 1822.888486209)


def test_get_period():
    atomic_number = 13
    period = get_period(atomic_number)
    assert period == 3  # Aluminium Period
