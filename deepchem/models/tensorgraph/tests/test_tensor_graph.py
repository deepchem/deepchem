import unittest

import numpy as np
import os
from nose.tools import assert_true

import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.data.datasets import Databag
from deepchem.models.tensorgraph.layers import Dense, SoftMaxCrossEntropy, ReduceMean, SoftMax
from deepchem.models.tensorgraph.layers import Feature, Label
from deepchem.models.tensorgraph.layers import ReduceSquareDifference
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class TestTensorGraph(unittest.TestCase):
  """
  Test that graph topologies work correctly.
  """

  def test_single_task_classifier(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(learning_rate=0.1)
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=10)
    prediction = np.squeeze(tg.predict_proba_on_batch(X))
    assert_true(np.all(np.isclose(prediction, y, atol=0.2)))

  def test_multi_task_classifier(self):
    n_data_points = 20
    n_features = 2

    X = np.random.rand(n_data_points, n_features)
    y1 = np.array([[0, 1] for x in range(n_data_points)])
    y2 = np.array([[1, 0] for x in range(n_data_points)])
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1), NumpyDataset(y2)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    entropies = []
    for i in range(2):
      label = Label(shape=(None, 2))
      dense = Dense(out_channels=2, in_layers=[features])
      output = SoftMax(in_layers=[dense])
      smce = SoftMaxCrossEntropy(in_layers=[label, dense])

      entropies.append(smce)
      outputs.append(output)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=entropies)

    tg = dc.models.TensorGraph(learning_rate=0.1)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
      databag.iterbatches(
        epochs=100, batch_size=tg.batch_size, pad_batches=True))
    prediction = tg.predict_proba_on_generator(databag.iterbatches())
    for i in range(2):
      y_real = ys[i].X
      y_pred = prediction[:, i, :]
      assert_true(np.all(np.isclose(y_pred, y_real, atol=0.2)))

  def test_single_task_regressor(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [0.5 for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=1, in_layers=[features])
    label = Label(shape=(None, 1))
    loss = ReduceSquareDifference(in_layers=[dense, label])
    tg = dc.models.TensorGraph(learning_rate=0.1)
    tg.add_output(dense)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=10)
    prediction = np.squeeze(tg.predict_proba_on_batch(X))
    assert_true(np.all(np.isclose(prediction, y, atol=1.5)))

  def test_multi_task_regressor(self):
    n_data_points = 20
    n_features = 2

    X = np.random.rand(n_data_points, n_features)
    y1 = np.expand_dims(np.array([0.5 for x in range(n_data_points)]), axis=-1)
    y2 = np.expand_dims(np.array([-0.5 for x in range(n_data_points)]), axis=-1)
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1), NumpyDataset(y2)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    losses = []
    for i in range(2):
      label = Label(shape=(None, 1))
      dense = Dense(out_channels=1, in_layers=[features])
      loss = ReduceSquareDifference(in_layers=[dense, label])

      outputs.append(dense)
      losses.append(loss)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=losses)

    tg = dc.models.TensorGraph(learning_rate=0.1)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
      databag.iterbatches(
        epochs=200, batch_size=tg.batch_size, pad_batches=True))
    prediction = tg.predict_proba_on_generator(databag.iterbatches())
    for i in range(2):
      y_real = ys[i].X
      y_pred = prediction[:, i, :]
      assert_true(np.all(np.isclose(y_pred, y_real, atol=1.5)))

  def test_no_queue(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(learning_rate=1.0, use_queue=False)
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=10)
    prediction = np.squeeze(tg.predict_proba_on_batch(X))
    assert_true(np.all(np.isclose(prediction, y, atol=0.2)))

  def test_tensorboard(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(
      tensorboard=True,
      tensorboard_log_frequency=1,
      learning_rate=0.1,
      model_dir='/tmp/tensorgraph')
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=10)
    files_in_dir = os.listdir(tg.model_dir)
    event_file = list(filter(lambda x: x.startswith("events"), files_in_dir))
    assert_true(len(event_file) > 0)
    event_file = os.path.join(tg.model_dir, event_file[0])
    file_size = os.stat(event_file).st_size
    assert_true(file_size > 0)

  def test_save_load(self):
    n_data_points = 20
    n_features = 2
    X = np.random.rand(n_data_points, n_features)
    y = [[0, 1] for x in range(n_data_points)]
    dataset = NumpyDataset(X, y)
    features = Feature(shape=(None, n_features))
    dense = Dense(out_channels=2, in_layers=[features])
    output = SoftMax(in_layers=[dense])
    label = Label(shape=(None, 2))
    smce = SoftMaxCrossEntropy(in_layers=[label, dense])
    loss = ReduceMean(in_layers=[smce])
    tg = dc.models.TensorGraph(learning_rate=0.1)
    tg.add_output(output)
    tg.set_loss(loss)
    tg.fit(dataset, nb_epoch=1)
    prediction = np.squeeze(tg.predict_proba_on_batch(X))
    tg.save()

    tg1 = TensorGraph.load_from_dir(tg.model_dir)
    prediction2 = np.squeeze(tg1.predict_proba_on_batch(X))
    assert_true(np.all(np.isclose(prediction, prediction2, atol=0.01)))
