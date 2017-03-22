import numpy as np
import unittest
import deepchem as dc
from deepchem.models.tf_new_models.tensor_graph import TensorGraph, LossLayer, Flatten
from deepchem.models.tf_new_models.tensor_graph import Input, Dense


class TestTensorGraph(unittest.TestCase):
  """
  Test that graph topologies work correctly.
  """

  def test_graph_save(self):
    n_samples = 10
    n_features = 11
    n_tasks = 1
    batch_size = 10
    X = np.random.rand(batch_size, n_samples, n_features)
    y = np.ones(shape=(n_samples, n_tasks))
    ids = np.arange(n_samples)

    dataset = dc.data.NumpyDataset(X, y, None, ids)
    g = TensorGraph()

    inLayer = Input(t_shape=(None, n_samples, n_features))
    g.add_layer(inLayer)

    flatten = Flatten()
    g.add_layer(flatten, parents=[inLayer])

    dense = Dense(out_channels=1)
    g.add_layer(dense, parents=[flatten])

    label_out = Input(t_shape=(None, 1))
    g.add_layer(label_out)

    loss = LossLayer()
    g.add_layer(loss, parents=[dense, label_out])

    g.features = inLayer.out_tensor
    g.labels = label_out.out_tensor
    g.loss = loss.out_tensor
    g.outputs = dense.out_tensor

    g.fit(dataset)
    print(g.predict(dataset.X))

