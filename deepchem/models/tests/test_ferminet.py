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
  molecule = Ferminet(h2_molecule, spin=0, charge=0, seed=0, batch_no=1)
  molecule.initialize_electrons()

  fermi = FerminetModel(torch.from_numpy(molecule.molecule.x),
                        spin=(molecule.up_spin, molecule.down_spin),
                        nuclear_charge=torch.from_numpy(molecule.charge),
                        inter_atom=torch.from_numpy(molecule.inter_atom))

  potential = fermi.calculate_potential()
  assert torch.allclose(potential, torch.tensor([-40.5568845023]))

  fermi.forward()

  assert torch.allclose(
      fermi.one_up,
      torch.Tensor([[[
          0.035281046919353284, 0.008003144167344467, 0.019574759682114785,
          0.041133609188854975
      ],
                     [
                         0.035281046919353284, 0.008003144167344467,
                         -0.7284252403178852, 0.7293230651230344
                     ]]]))

  assert torch.allclose(
      fermi.one_down,
      torch.Tensor([[[
          0.044817863984029156, 0.03735115980299935, 0.7284544424024718,
          0.7307869899817704
      ],
                     [
                         0.044817863984029156, 0.03735115980299935,
                         -0.019545557597528185, 0.06152868349411048
                     ]]]))
  assert fermi.one_electron_distance.size(
  )[0] == 2 and fermi.one_electron_distance.size()[1] == 2
  assert torch.allclose(
      fermi.two_up,
      torch.Tensor([[[0.0, 0.0, 0.0, 0.0],
                     [
                         0.009536817064675872, 0.02934801563565488,
                         0.7088796827203571, 0.7095510280981839
                     ]]]))
  assert torch.allclose(
      fermi.two_down,
      torch.Tensor([[[
          -0.009536817064675872, -0.02934801563565488, -0.7088796827203571,
          0.7095510280981839
      ], [0.0, 0.0, 0.0, 0.0]]]))
  assert fermi.two_electron_distance.size(
  )[0] == 2 and fermi.two_electron_distance.size()[1] == 2

  # potential energy test
  # potential = molecule.calculate_potential()
  # assert np.allclose(potential, [-40.5568845023])

  # ionic charge initialization test
  # ion = [['C', [0, 0, 0]], ['O', [0, 3, 0]], ['O', [1, -1, 0]],
  #        ['O', [-1, -1, 0]]]  # Test ionic molecule
  # ionic_molecule = Ferminet(ion, spin=1, charge=-2, seed=0, batch_no=1)
  # _, _, _, _ = ionic_molecule.prepare_input_stream()

  # assert (ionic_molecule.electron_no == np.array([[6], [8], [9], [9]])).all()
