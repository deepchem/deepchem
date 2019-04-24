import unittest
import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context


class TestKerasModel(unittest.TestCase):

  def test_overfit_graph_model(self):
    """Test fitting a KerasModel defined as a graph."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features).astype(np.float32)
    y = np.expand_dims(X[:, 0] > X[:, 1], 1).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    inputs = tf.keras.Input(shape=(n_features,))
    hidden = tf.keras.layers.Dense(10, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def loss_fn(inputs, labels, weights):
      return tf.reduce_mean(
          tf.keras.metrics.binary_crossentropy(labels[0],
                                               keras_model(inputs[0])))

    model = dc.models.KerasModel(keras_model, loss_fn, learning_rate=0.005)
    model.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.all(np.isclose(prediction, y.flatten(), atol=0.4))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = model.evaluate(dataset, [metric])
    assert scores[metric.name] > .9

  def test_overfit_graph_model_eager(self):
    """Test fitting a KerasModel defined as a graph, in eager mode."""
    with context.eager_mode():
      self.test_overfit_graph_model()

  def test_overfit_sequential_model(self):
    """Test fitting a KerasModel defined as a sequential model."""
    n_data_points = 10
    n_features = 2
    X = np.random.rand(n_data_points, n_features).astype(np.float32)
    y = np.expand_dims(X[:, 0] > X[:, 1], 1).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    def loss_fn(inputs, labels, weights):
      return tf.reduce_mean(
          tf.keras.metrics.binary_crossentropy(labels[0],
                                               keras_model(inputs[0])))

    model = dc.models.KerasModel(keras_model, loss_fn, learning_rate=0.005)
    model.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(model.predict_on_batch(X))
    assert np.all(np.isclose(prediction, y.flatten(), atol=0.4))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = model.evaluate(dataset, [metric])
    assert scores[metric.name] > .9

  def test_overfit_sequential_model_eager(self):
    """Test fitting a KerasModel defined as a sequential model, in eager mode."""
    with context.eager_mode():
      self.test_overfit_sequential_model()
