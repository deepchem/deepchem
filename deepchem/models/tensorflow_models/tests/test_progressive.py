"""
Sanity tests on progressive models.
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


class TestProgressive(test_util.TensorFlowTestCase):
  """
  Test that progressive models satisfy basic sanity checks. 
  """

  def setUp(self):
    super(TestProgressive, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_construction(self):
    """Test that progressive models can be constructed without crash."""
    prog_model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks=1,
        n_features=100,
        alpha_init_stddevs=[.08],
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
        batch_size=200)

  def test_fit(self):
    """Test that progressive models can fit without crash."""
    n_tasks = 2
    n_samples = 10
    n_features = 100
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    prog_model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks=n_tasks,
        n_features=n_features,
        alpha_init_stddevs=[.08],
        dropouts=[0.],
        learning_rate=0.003,
        weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
        batch_size=2)

    prog_model.fit(dataset)

  def test_fit_lateral(self):
    """Test that multilayer model fits correctly.

    Lateral connections and adapters are only added for multilayer models. Test
    that fit functions with multilayer models.
    """
    n_tasks = 2
    n_samples = 10
    n_features = 100
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    n_layers = 3
    prog_model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks=n_tasks,
        n_features=n_features,
        alpha_init_stddevs=[.08] * n_layers,
        layer_sizes=[100] * n_layers,
        weight_init_stddevs=[.02] * n_layers,
        bias_init_consts=[1.] * n_layers,
        dropouts=[0.] * n_layers,
        learning_rate=0.003,
        batch_size=2)

    prog_model.fit(dataset)

  def test_fit_lateral_multi(self):
    """Test that multilayer model fits correctly.

    Test multilayer model with multiple tasks (> 2) to verify that lateral
    connections of growing size work correctly.
    """
    n_tasks = 3
    n_samples = 10
    n_features = 100
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    n_layers = 3
    prog_model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks=n_tasks,
        n_features=n_features,
        alpha_init_stddevs=[.08] * n_layers,
        layer_sizes=[100] * n_layers,
        weight_init_stddevs=[.02] * n_layers,
        bias_init_consts=[1.] * n_layers,
        dropouts=[0.] * n_layers,
        learning_rate=0.003,
        batch_size=2)

    prog_model.fit(dataset)

  def test_frozen_weights(self):
    """Test that fitting one task doesn't change predictions of another.
    
    Tests that weights are frozen when training different tasks.
    """
    n_tasks = 2
    n_samples = 10
    n_features = 100
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    n_layers = 3
    prog_model = dc.models.ProgressiveMultitaskRegressor(
        n_tasks=n_tasks,
        n_features=n_features,
        alpha_init_stddevs=[.08] * n_layers,
        layer_sizes=[100] * n_layers,
        weight_init_stddevs=[.02] * n_layers,
        bias_init_consts=[1.] * n_layers,
        dropouts=[0.] * n_layers,
        learning_rate=0.003,
        batch_size=2)

    # Fit just on task zero 
    # Notice that we keep the session open
    prog_model.fit(dataset, tasks=[0], close_session=False)
    y_pred_task_zero = prog_model.predict(dataset)[:, 0]

    # Fit on task one
    prog_model.fit(dataset, tasks=[1])
    y_pred_task_zero_after = prog_model.predict(dataset)[:, 0]

    # The predictions for task zero should not change after training
    # on task one. 
    np.testing.assert_allclose(y_pred_task_zero, y_pred_task_zero_after)
