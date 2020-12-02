import numpy as np
from deepchem.feat.material_featurizers.lcnn_featurizer import LCNNFeaturizer
from data.lcnn_test_data import primitive_cell, structure, check_edges, check_feature


def test_LCNNFeaturizer():
  featuriser = LCNNFeaturizer(np.around(6.00), primitive_cell)
  data = featuriser._featurize(structure)
  assert np.all(data['X_Sites'] == np.array(check_feature))
  assert np.all(data['X_NSs'] == np.array(check_edges))
  assert data['X_Sites'].shape == (4, 3)
  assert data['X_NSs'].shape == (1, 4, 6, 19)
