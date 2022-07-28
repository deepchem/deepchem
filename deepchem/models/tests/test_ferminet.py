"""
Test for Ferminet Model.
"""

import numpy as np
try:
  from deepchem.models.torch_models.ferminet import Ferminet
except ModuleNotFoundError:
  pass


def test_prepare_input_stream():
  # test for the prepare_input_stream function of Ferminet class

  h2_molecule = [['[H]', [0, 0, 0]], ['[H]', [0, 0, 0.748]]]
  molecule = Ferminet(h2_molecule)
  molecule.prepare_input_stream(seed_no=0)

  assert molecule.one_electron_vector == np.array(
      [[[0.03528105, 0.00800314, 0.01957476],
        [0.03528105, 0.00800314, -0.72842524]],
       [[0.04481786, 0.03735116, 0.72845444],
        [0.04481786, 0.03735116, -0.01954556]]])
  assert molecule.one_electron_distance == np.array([[0.04113361, 0.72932307],
                                                     [0.73078699, 0.06152868]])
  assert molecule.two_electron_vector == np.array(
      [[[0., 0., 0.], [0.00953682, 0.02934802,
                       0.70887968]][[-0.00953682, -0.02934802,
                                     -0.70887968][0., 0., 0.]]])
  assert molecule.two_electron_distance == np.array([[0., 0.70955103],
                                                     [0.70955103, 0.]])
