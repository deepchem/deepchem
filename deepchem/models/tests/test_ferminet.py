"""
Test for Ferminet Model.
"""

import pytest
from deepchem.models.torch_models.ferminet import Ferminet
import numpy as np


@pytest.mark.torch
def test_prepare_input_stream():
  mol = Ferminet([['H', [0, 0, -1]], ['H', [0, 0, 1]]])
  mol.prepare_input_stream()

  assert mol.one_electron == np.array(
      [[0, 0, 1],
       [0, 0,
        -1]])  # TODO replace this value with actual value by setting a seed
  assert mol.two_electron == np.array(
      [[0, 0, 1],
       [0, 0,
        -1]])  # TODO replace this value with actual value by setting a seed
