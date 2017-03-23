import unittest

import numpy as np

import deepchem as dc
from models.tensorgraph.tensor_graph import Input, Dense
from models.tensorgraph.tensor_graph import TensorGraph, LossLayer, Flatten


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
    g = TensorGraph(model_dir='/tmp/tmpss5_ki5_')

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

    g.add_feature(inLayer)
    g.add_label(label_out)
    g.set_loss(loss)
    g.add_output(dense)


    g.fit(dataset, nb_epoch=100)
    g.save()
    g1 = TensorGraph.load_from_dir('/tmp/tmpss5_ki5_')
    print(g1)
    print(g1.predict_on_batch(X))
    #print(g.predict(dataset.X))

