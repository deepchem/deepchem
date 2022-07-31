"""
Test for Ferminet Model.
"""

import pytest
import numpy as np
try:
  from deepchem.models.torch_models.ferminet import Ferminet
except ModuleNotFoundError:
  pass


@pytest.mark.torch
def test_prepare_input_stream():
  # test for the prepare_input_stream function of Ferminet class

  h2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
  molecule = Ferminet(h2_molecule, seed_no=0, batch_number=1)
  molecule.prepare_input_stream()

  assert np.allclose(molecule.one_electron_vector, np.array(
      [[[0.035281046919353284, 0.008003144167344467, 0.019574759682114785],
        [0.035281046919353284, 0.008003144167344467, -0.7284252403178852]],
       [[0.044817863984029156, 0.03735115980299935, 0.7284544424024718],
        [0.044817863984029156, 0.03735115980299935,
         -0.019545557597528185]]]))
  assert np.allclose(molecule.one_electron_distance == np.array(
      [[0.041133609188854975, 0.7293230651230344],
       [0.7307869899817704, 0.06152868349411048]]))
  assert np.allclose(molecule.two_electron_vector == np.array(
      [[[0.0, 0.0, 0.0],
        [0.009536817064675872, 0.02934801563565488, 0.7088796827203571]],
       [[-0.009536817064675872, -0.02934801563565488, -0.7088796827203571],
        [0.0, 0.0, 0.0]]]))
  assert np.allclose(molecule.two_electron_distance == np.array([[0.0, 0.7095510280981839],
                                                      [0.7095510280981839,
                                                       0.0]]))
