import unittest

import deepchem
import numpy as np
from deepchem.models import TensorGraph


class TestSaScoreModel(unittest.TestCase):

  def test_save_load(self):
    """Test SaScoreModel anc be saved and loaded"""
    n_samples = 10
    n_features = 3
    n_tasks = 1

    # Create a dataset and an input function for processing it.

    np.random.seed(123)
    X = np.random.rand(n_samples, 2, n_features)
    y = np.zeros((n_samples, n_tasks))
    dataset = deepchem.data.NumpyDataset(X, y)

    model = deepchem.models.ScScoreModel(n_features, dropouts=0)

    model.fit(dataset, nb_epoch=1)
    pred1 = model.predict(dataset)

    model.save()
    model = TensorGraph.load_from_dir(model.model_dir)

    pred2 = model.predict(dataset)
    for m1, m2 in zip(pred1, pred2):
      self.assertTrue(np.all(m1 == m2))
