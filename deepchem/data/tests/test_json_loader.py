"""
Tests for JsonLoader class.
"""

import os
import unittest
import tempfile
import shutil
import numpy as np
import deepchem as dc
from deepchem.data.data_loader import JsonLoader
from deepchem.feat.materials_featurizers import SineCoulombMatrix


class TestJsonLoader(unittest.TestCase):
  """
  Test JsonLoader
  """

  def setUp(self):
    super(TestJsonLoader, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_json_loader(self):
    input_file = os.path.join(self.current_dir, 'perov_test.json')
    featurizer = SineCoulombMatrix(max_atoms=5)
    loader = JsonLoader(
        tasks=['e_form'],
        json_fields={"structure": dict},
        id_field='formula',
        featurizer=featurizer)
    dataset = loader.create_dataset(input_file, shard_size=1)

    a = [4625.32086965, 6585.20209678, 61.00680193, 48.72230922, 48.72230922]

    assert dataset.X.shape == (5, 1, 5)
    assert np.allclose(dataset.X[0][0], a, atol=.5)
