import os
import unittest
import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.keras.layers import Input, Dense
from deepchem.models.losses import L2Loss


class MLP(dc.models.KerasModel):

  def __init__(self, n_tasks=1, feature_dim=100, hidden_layer_size=64,
               **kwargs):
    self.feature_dim = feature_dim
    self.hidden_layer_size = hidden_layer_size
    self.n_tasks = n_tasks

    model, loss, output_types = self._build_graph()
    super(MLP, self).__init__(
        model=model, loss=loss, output_types=output_types, **kwargs)

  def _build_graph(self):
    inputs = Input(dtype=tf.float32, shape=(self.feature_dim,), name="Input")
    out1 = Dense(units=self.hidden_layer_size, activation='relu')(inputs)

    final = Dense(units=self.n_tasks, activation='sigmoid')(out1)
    outputs = [final]
    output_types = ['prediction']
    loss = dc.models.losses.BinaryCrossEntropy()

    model = tf.keras.Model(inputs=[inputs], outputs=outputs)
    return model, loss, output_types


class TestPretrained(unittest.TestCase):

  def setUp(self):
    model_dir = "./MLP/"
    self.feature_dim = 2
    self.hidden_layer_size = 10
    data_points = 10

    X = np.random.randn(data_points, self.feature_dim)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)

    self.dataset = dc.data.NumpyDataset(X, y)

    model = MLP(
        hidden_layer_size=self.hidden_layer_size,
        feature_dim=self.feature_dim,
        model_dir=model_dir,
        batch_size=10)
    model.fit(self.dataset, nb_epoch=1000)
    predictions = np.squeeze(model.predict_on_batch(self.dataset.X))
    np.testing.assert_array_almost_equal(self.dataset.y, np.round(predictions))

  def test_load_from_pretrained_graph_mode(self):
    """Tests loading pretrained model in graph mode."""
    source_model = MLP(
        model_dir="./MLP/",
        feature_dim=self.feature_dim,
        hidden_layer_size=self.hidden_layer_size)

    dest_model = MLP(
        feature_dim=self.feature_dim,
        hidden_layer_size=self.hidden_layer_size,
        n_tasks=10)

    assignment_map = dict()
    dest_vars = dest_model.model.trainable_variables[:-2]

    for idx, dest_var in enumerate(dest_vars):
      source_var = source_model.model.trainable_variables[idx]
      assignment_map[source_var] = dest_var

    dest_model.load_from_pretrained(
        source_model=source_model,
        assignment_map=assignment_map,
        include_top=False)

    for source_var, dest_var in assignment_map.items():
      np.testing.assert_array_almost_equal(
          source_var.eval(session=dest_model.session),
          dest_var.eval(session=dest_model.session))

  def test_load_from_pretrained_eager(self):
    """Tests loading pretrained model in eager execution mode."""
    with context.eager_mode():
      source_model = MLP(
          model_dir="./MLP/",
          feature_dim=self.feature_dim,
          hidden_layer_size=self.hidden_layer_size)

      dest_model = MLP(
          feature_dim=self.feature_dim,
          hidden_layer_size=self.hidden_layer_size,
          n_tasks=10)

      assignment_map = dict()
      dest_vars = dest_model.model.trainable_variables[:-2]

      for idx, dest_var in enumerate(dest_vars):
        source_var = source_model.model.trainable_variables[idx]
        assignment_map[source_var] = dest_var

      dest_model.load_from_pretrained(
          source_model=source_model, assignment_map=assignment_map)

      for source_var, dest_var in assignment_map.items():
        np.testing.assert_array_almost_equal(source_var.numpy(),
                                             dest_var.numpy())

  def test_restore_equivalency(self):
    source_model = MLP(
        model_dir="./MLP/",
        feature_dim=self.feature_dim,
        hidden_layer_size=self.hidden_layer_size)

    dest_model = MLP(
        feature_dim=self.feature_dim, hidden_layer_size=self.hidden_layer_size)

    dest_model.load_from_pretrained(
        source_model=source_model, assignment_map=None, include_top=True)

    predictions = np.squeeze(dest_model.predict_on_batch(self.dataset.X))

    np.testing.assert_array_almost_equal(self.dataset.y, np.round(predictions))
