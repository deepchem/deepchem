import os
import json
import numpy as np
from deepchem.feat.material_featurizers.lcnn_featurizer import LCNNFeaturizer
try:
  from pymatgen import Structure
except:
  raise ImportError("This class requires pymatgen to be installed.")


def test_LCNNFeaturizer():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  structure_file = os.path.join(current_dir, 'data',
                                'test_platinum_adsorption.json')
  with open(structure_file, 'r') as f:
    test_data = json.load(f)
    test_data["primitive_cell"]["structure"] = Structure.from_dict(
        test_data["primitive_cell"]["structure"])
    test_data["data point"] = Structure.from_dict(test_data["data point"])

    featuriser = LCNNFeaturizer(**test_data["primitive_cell"])
    data = featuriser._featurize(test_data["data point"])
    assert np.all(data.node_features == np.array(test_data["node_feature"]))
    assert np.all(data.edge_index == test_data["edges"])
    assert data.node_features.shape == (4, 3)
    assert data.edge_index.shape == (2, 456)
