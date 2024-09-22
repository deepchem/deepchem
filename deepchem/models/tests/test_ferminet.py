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


@pytest.mark.torch
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


@pytest.mark.torch
def test_forward():
    FH_molecule = [['F', [0.424, 0.424, 0.23]], ['H', [0.4, 0.5, 0.5]]]
    # Testing ionic initialization
    mol = FerminetModel(FH_molecule, spin=1, ion_charge=-1)
    result = mol.model.forward(torch.from_numpy(mol.molecule.x))
    assert result.size() == torch.Size([8])


@pytest.mark.torch
def test_evaluate_hf_solution():
    # Test for the evaluate_hf_solution function of FerminetModel class
    H2_molecule = [['F', [0, 0, 0]], ['He', [0, 0, 0.748]]]
    mol = FerminetModel(H2_molecule, spin=1, ion_charge=0)
    electron_coordinates = np.random.rand(10, 11, 3)
    spin_up_orbitals, spin_down_orbitals = mol.evaluate_hf(electron_coordinates)
    # The solution should be of the shape (number of electrons, number of electrons)
    assert np.shape(spin_up_orbitals) == (10, 6, 6)
    assert np.shape(spin_down_orbitals) == (10, 5, 5)


@pytest.mark.torch
def test_FerminetModel_pretrain():
    # Test for the init function of FerminetModel class
    H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    # Testing ionic initialization
    mol = FerminetModel(H2_molecule, spin=0, ion_charge=0)
    mol.train(nb_epoch=3)
    assert mol.loss_value <= torch.tensor(1.0)


@pytest.mark.torch
def test_FerminetModel_energy():
    # Test for the init function of FerminetModel class
    H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    # Testing ionic initialization
    mol = FerminetModel(H2_molecule, spin=0, ion_charge=0)
    mol.train(nb_epoch=5)
    energy = mol.model.calculate_electron_electron(
    ) - mol.model.calculate_electron_nuclear(
    ) + mol.model.nuclear_nuclear_potential + mol.model.calculate_kinetic_energy(
    )
    mean_energy = torch.mean(energy)
    assert mean_energy <= torch.tensor(1.0)


@pytest.mark.torch
def test_FerminetModel_train():
    # Test for the init function of FerminetModel class
    H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    # Testing ionic initialization
    mol = FerminetModel(H2_molecule, spin=0, ion_charge=0)
    mol.train(nb_epoch=10)
    mol.prepare_train()
    mol.train(nb_epoch=10)
    assert mol.final_energy <= torch.tensor(0.0)


@pytest.mark.torch
def test_FerminetModel_ion_train():
    # Test for the init function of FerminetModel class
    H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    # Testing ionic initialization
    mol = FerminetModel(H2_molecule, spin=1, ion_charge=-1)
    mol.train(nb_epoch=10)
    mol.prepare_train()
    mol.train(nb_epoch=10)
    assert (mol.final_energy >= torch.tensor(-0.2)) and (mol.final_energy <=
                                                         torch.tensor(1.0))
