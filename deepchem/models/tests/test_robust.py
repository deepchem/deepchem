import numpy as np
import tensorflow as tf
import deepchem as dc


def test_singletask_robust_multitask_classification():
  """Test robust multitask singletask classification."""
  n_tasks = 1
  n_samples = 10
  n_features = 3
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  dataset = dc.data.NumpyDataset(X, y, w, ids)

  classification_metric = dc.metrics.Metric(
      dc.metrics.accuracy_score, task_averager=np.mean)
  model = dc.models.RobustMultitaskClassifier(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples)

  # Fit trained model
  model.fit(dataset, nb_epoch=1)


def test_singletask_robust_multitask_regression():
  """Test singletask robust multitask regression."""
  np.random.seed(123)
  tf.random.set_seed(123)
  n_tasks = 1
  n_samples = 10
  n_features = 3
  n_classes = 2

  # Generate dummy dataset
  np.random.seed(123)
  ids = np.arange(n_samples)
  X = np.random.rand(n_samples, n_features)
  y = np.zeros((n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  regression_metric = dc.metrics.Metric(
      dc.metrics.mean_squared_error, task_averager=np.mean, mode="regression")
  model = dc.models.RobustMultitaskRegressor(
      n_tasks,
      n_features,
      layer_sizes=[50],
      bypass_layer_sizes=[10],
      dropouts=[0.],
      learning_rate=0.003,
      weight_init_stddevs=[.1],
      batch_size=n_samples)

  # Fit trained model
  model.fit(dataset, nb_epoch=1)
