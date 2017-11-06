import unittest
import numpy as np
import deepchem as dc
from deepchem.models.tensorgraph.layers import Dense
from deepchem.models.tensorgraph.layers import SoftMax
from nose.tools import assert_true


class TestSequential(unittest.TestCase):
  """
  Test that sequential graphs work correctly.
  """

  def test_single_task_classifier(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = dc.data.NumpyDataset(X, y)
    model = dc.models.Sequential(learning_rate=0.01)
    model.add(Dense(out_channels=2))
    model.add(SoftMax())
    model.fit(dataset, loss="binary_crossentropy", nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert_true(np.all(np.isclose(prediction, y, atol=0.4)))
