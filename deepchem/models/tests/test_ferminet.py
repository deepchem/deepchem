"""
Test for Ferminet Model.
"""

import pytest
import numpy as np
try:
    from deepchem.models.torch_models.ferminet import FerminetModel
    import torch
    # When pytest runs without pytorch in the environment (ex: as in tensorflow workflow),
    # the above import raises a ModuleNotFoundError. It is safe to ignore it
    # since the below tests only run in an environment with pytorch installed.
except ModuleNotFoundError:
    pass


@pytest.mark.dqc
def test_FerminetModel():
    # Test for the init function of FerminetModel class
    FH_molecule = [['F', [0, 0, 0]], ['H', [0, 0.5, 0.5]]]
    # Testing ionic initialization
    mol = FerminetModel(FH_molecule, spin=1, ion_charge=-1)
    assert (mol.electron_no == np.array([[10], [1]])).all()
    # Testing whether error throws up when spin is wrong
    with pytest.raises(ValueError):
        FerminetModel(FH_molecule, spin=0, ion_charge=-1)
    # Testing the spin values
    Li_atom = [['Li', [0, 0, 0]]]
    mol = FerminetModel(Li_atom, spin=1, ion_charge=0)
    assert mol.up_spin == 2 and mol.down_spin == 1


@pytest.mark.dqc
def test_forward():
    FH_molecule = [['F', [0.424, 0.424, 0.23]], ['H', [0.4, 0.5, 0.5]]]
    # Testing ionic initialization
    mol = FerminetModel(FH_molecule, spin=1, ion_charge=-1)
    result = mol.model.forward(mol.molecule.x)
    assert result.size() == torch.Size([8])
