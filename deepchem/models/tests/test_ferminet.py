"""
Test for Ferminet Model.
"""

import pytest
import numpy as np

try:
  import torch
  from deepchem.models.torch_models.ferminet import Ferminet
  from deepchem.models.torch_models.ferminet import FerminetModel
except ModuleNotFoundError:
  pass


@pytest.mark.torch
def test_prepare_input_stream():
  # test for the prepare_input_stream function of Ferminet class
  h2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
  molecule = FerminetModel(h2_molecule, spin=0, charge=0, seed=0, batch_no=10)
  molecule.prepare_hf_solution()
  assert np.shape(molecule.mo_values) == (2, 2)
  # input = torch.tensor([[0, 0, 0], [0, 0, 0.748]], requires_grad=True)
  # fermi = Ferminet(input,
  #               spin=(molecule.up_spin, molecule.down_spin),
  #               nuclear_charge=torch.from_numpy(molecule.charge),
  #               inter_atom=torch.from_numpy(molecule.inter_atom))
  molecule.fit()

  # molecule_input = torch.tensor(molecule.molecule.x, requires_grad=True)
  # log_psi = fermi.forward(molecule_input.to(device))
  # fermi.local_energy()
  # print(fermi.loss(log_psi, torch.tensor([-40.5568845023])))
  # potential = fermi.calculate_potential()
  # assert torch.allclose(potential, torch.tensor([-40.5568845023]))

  # TODO: add the local_energy test after pretraining is done

  # potential energy test
  # potential = molecule.calculate_potential()
  # assert np.allclose(potential, [-40.5568845023])

  # ionic charge initialization test
  # ion = [['C', [0, 0, 0]], ['O', [0, 3, 0]], ['O', [1, -1, 0]],
  #        ['O', [-1, -1, 0]]]  # Test ionic molecule
  # ionic_molecule = Ferminet(ion, spin=1, charge=-2, seed=0, batch_no=1)
  # _, _, _, _ = ionic_molecule.prepare_input_stream()

  # assert (ionic_molecule.electron_no == np.array([[6], [8], [9], [9]])).all()
