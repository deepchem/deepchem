"""
Tests for JsonLoader class.
"""

import os
import numpy as np
from deepchem.data.data_loader import JsonLoader
from deepchem.feat import SineCoulombMatrix


def test_json_loader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'inorganic_crystal_sample_data.json')
    featurizer = SineCoulombMatrix(max_atoms=5)
    loader = JsonLoader(tasks=['e_form'],
                        feature_field='structure',
                        id_field='formula',
                        label_field='e_form',
                        featurizer=featurizer)

    dataset = loader.create_dataset(input_file, shard_size=1)

    a = [4625.32086965, 6585.20209678, 61.00680193, 48.72230922, 48.72230922]

    assert dataset.X.shape == (5, 5)
    assert np.allclose(dataset.X[0], a, atol=.5)

    dataset = loader.create_dataset(input_file, shard_size=None)
    assert dataset.X.shape == (5, 5)

    dataset = loader.create_dataset([input_file, input_file], shard_size=5)
    assert dataset.X.shape == (10, 5)
