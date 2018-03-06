import unittest
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.data import NumpyDataset


class TestEstimators(unittest.TestCase):
  """
  Test converting TensorGraphs to Estimators.
  """

  def test_multi_task_classifier(self):
    """Test creating an Estimator from a MultiTaskClassifier."""
    n_samples = 10
    n_features = 3
    n_tasks = 2

    # Create a dataset and an input function for processing it.

    np.random.seed(123)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y)

    def input_fn(epochs):
      x, y, weights = dataset.make_iterator(
          batch_size=n_samples, epochs=epochs).get_next()
      return {'x': x, 'weights': weights}, y

    # Create a TensorGraph model.

    model = dc.models.MultiTaskClassifier(n_tasks, n_features, dropouts=0)

    # Create an estimator from it.

    x_col = tf.feature_column.numeric_column('x', shape=(n_features,))
    weight_col = tf.feature_column.numeric_column('weights', shape=(n_tasks,))

    def accuracy(labels, predictions, weights):
      return tf.metrics.accuracy(labels, tf.round(predictions), weights)

    metrics = {'accuracy': accuracy}
    estimator = model.make_estimator(
        feature_columns=[x_col], weight_column=weight_col, metrics=metrics)

    # Train the model.

    estimator.train(input_fn=lambda: input_fn(100))

    # Evaluate the model.

    results = estimator.evaluate(input_fn=lambda: input_fn(1))
    assert results['loss'] < 1e-4
    assert results['accuracy'] > 0.9

  def test_multi_task_regressor(self):
    """Test creating an Estimator from a MultiTaskRegressor."""
    n_samples = 10
    n_features = 3
    n_tasks = 2

    # Create a dataset and an input function for processing it.

    np.random.seed(123)
    X = np.random.rand(n_samples, n_features)
    y = np.zeros((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y)

    def input_fn(epochs):
      x, y, weights = dataset.make_iterator(
          batch_size=n_samples, epochs=epochs).get_next()
      return {'x': x, 'weights': weights}, y

    # Create a TensorGraph model.

    model = dc.models.MultiTaskRegressor(n_tasks, n_features, dropouts=0)

    # Create an estimator from it.

    x_col = tf.feature_column.numeric_column('x', shape=(n_features,))
    weight_col = tf.feature_column.numeric_column('weights', shape=(n_tasks,))
    metrics = {'error': tf.metrics.mean_absolute_error}
    estimator = model.make_estimator(
        feature_columns=[x_col], weight_column=weight_col, metrics=metrics)

    # Train the model.

    estimator.train(input_fn=lambda: input_fn(100))

    # Evaluate the model.

    results = estimator.evaluate(input_fn=lambda: input_fn(1))
    assert results['loss'] < 1e-3
    assert results['error'] < 0.1
