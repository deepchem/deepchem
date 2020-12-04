import os
import json
import numpy as np
from deepchem.feat.material_featurizers.lcnn_featurizer import LCNNFeaturizer


def test_LCNNFeaturizer():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  strucutre_file = os.path.join(current_dir, 'data',
                                'platinum_absorption_strucutre.json')
  with open(strucutre_file, 'r') as f:
    test_data = json.load(f)
    featuriser = LCNNFeaturizer(np.around(6.00), test_data["primitive_cell"])
    data = featuriser._featurize(test_data["structure"])
    assert np.all(data['X_Sites'] == np.array(test_data["node_feature"]))
    assert np.all(data['X_NSs'] == test_data["edges"])
    assert data['X_Sites'].shape == (4, 3)
    assert data['X_NSs'].shape == (1, 4, 6, 19)
