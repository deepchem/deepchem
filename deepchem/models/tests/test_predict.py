"""
Tests that deepchem models make deterministic predictions. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import tempfile
import numpy as np
import unittest
import sklearn
import shutil
import tensorflow as tf
import deepchem as dc
from tensorflow.python.framework import test_util
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


class TestPredict(test_util.TensorFlowTestCase):
  """
  Test that models make deterministic predictions 

  These tests guard against failures like having dropout turned on at
  test time.
  """

  def setUp(self):
    super(TestPredict, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_tf_progressive_regression_predict(self):
    """Test tf progressive multitask makes deterministic predictions."""
    np.random.seed(123)
    n_tasks = 9
    n_samples = 10
    n_features = 3
    n_classes = 2

    # Generate dummy dataset
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))

    dataset = dc.data.NumpyDataset(X, y, w, ids)

    regression_metric = dc.metrics.Metric(
        dc.metrics.mean_squared_error, task_averager=np.mean)
    model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks,
        n_features,
        layer_sizes=[50],
        bypass_layer_sizes=[10],
        dropouts=[.25],
        learning_rate=0.003,
        weight_init_stddevs=[.1],
        alpha_init_stddevs=[.02],
        batch_size=n_samples)

    # Fit trained model
    model.fit(dataset, nb_epoch=25)
    model.save()

    # Check same predictions are made. 
    y_pred_first = model.predict(dataset)
    y_pred_second = model.predict(dataset)
    np.testing.assert_allclose(y_pred_first, y_pred_second)
