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
    out1 = Dense(units=self.hidden_layer_size, activation=tf.nn.relu)(inputs)

    final = Dense(units=self.n_tasks)(out1)
    outputs = [final]
    output_types = ['prediction']
    loss = L2Loss()

    model = tf.keras.Model(inputs=[inputs], outputs=outputs)
    return model, loss, output_types


class TestPretrained(unittest.TestCase):

  def setUp(self):
    model_dir = "./MLP/"
    self.feature_dim = 2
    self.hidden_layer_size = 2
    data_points = 100

    X = np.random.randn(data_points, self.feature_dim)
    y = np.random.randn(data_points)

    dataset = dc.data.NumpyDataset(X, y)

    model = MLP(
        hidden_layer_size=self.hidden_layer_size,
        feature_dim=self.feature_dim,
        model_dir=model_dir,
        batch_size=10)
    model.fit(dataset, nb_epoch=100)

  def test_load_pretrained(self):
    source_model = MLP(
        model_dir="./MLP/",
        feature_dim=self.feature_dim,
        hidden_layer_size=self.hidden_layer_size)
    source_model.restore()

    dest_model = MLP(
        feature_dim=self.feature_dim,
        hidden_layer_size=self.hidden_layer_size,
        n_tasks=10)

    assignment_map = dict()
    dest_variables = dest_model.model.trainable_variables[:
                                                          -2]  #Excluding the last weight and bias

    for idx, variable in enumerate(dest_variables):
      source_variable = source_model.model.trainable_variables[idx]
      assignment_map[source_variable] = variable

    dest_model.load_pretrained(assignment_map=assignment_map)

    for var_old, var_new in assignment_map.items():
      val_old = dest_model.session.run(var_old)
      val_new = dest_model.session.run(var_new)

      np.testing.assert_array_almost_equal(val_old, val_new)
