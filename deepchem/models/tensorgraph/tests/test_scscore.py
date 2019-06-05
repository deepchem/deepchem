import unittest

import deepchem
import numpy as np
from deepchem.models import TensorGraph


class TestScScoreModel(unittest.TestCase):

  def test_overfit_scscore(self):
    """Test fitting to a small dataset"""
    n_samples = 10
    n_features = 3
    n_tasks = 1

    # Create a dataset and an input function for processing it.

    X = np.random.rand(n_samples, 2, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    dataset = deepchem.data.NumpyDataset(X, y)

    model = deepchem.models.ScScoreModel(n_features, dropouts=0)

    model.fit(dataset, nb_epoch=100)
    pred = model.predict(dataset)
    assert np.array_equal(y, pred[0] > pred[1])
