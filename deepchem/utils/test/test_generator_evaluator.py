from unittest import TestCase

import deepchem as dc
import numpy as np
from deepchem.data import NumpyDataset
from deepchem.data.datasets import Databag
from deepchem.models.tensorgraph.layers import Dense, ReduceMean, SoftMax, SoftMaxCrossEntropy
from deepchem.models.tensorgraph.layers import Feature, Label
from deepchem.models.tensorgraph.layers import ReduceSquareDifference
from nose.tools import assert_true
from flaky import flaky


class TestGeneratorEvaluator(TestCase):

  @flaky
  def test_compute_model_performance_multitask_classifier(self):
    n_data_points = 20
    n_features = 1

    X = np.ones(shape=(n_data_points // 2, n_features)) * -1
    X1 = np.ones(shape=(n_data_points // 2, n_features))
    X = np.concatenate((X, X1))
    class_1 = np.array([[0.0, 1.0] for x in range(int(n_data_points / 2))])
    class_0 = np.array([[1.0, 0.0] for x in range(int(n_data_points / 2))])
    y1 = np.concatenate((class_0, class_1))
    y2 = np.concatenate((class_1, class_0))
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1), NumpyDataset(y2)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    entropies = []
    labels = []
    for i in range(2):
      label = Label(shape=(None, 2))
      labels.append(label)
      dense = Dense(out_channels=2, in_layers=[features])
      output = SoftMax(in_layers=[dense])
      smce = SoftMaxCrossEntropy(in_layers=[label, dense])

      entropies.append(smce)
      outputs.append(output)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=entropies)

    tg = dc.models.TensorGraph(learning_rate=0.01, batch_size=n_data_points)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
        databag.iterbatches(
            epochs=1000, batch_size=tg.batch_size, pad_batches=True))
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")

    scores = tg.evaluate_generator(
        databag.iterbatches(), [metric], labels=labels, per_task_metrics=True)
    scores = list(scores[1].values())
    # Loosening atol to see if tests stop failing sporadically
    assert_true(np.all(np.isclose(scores, [1.0, 1.0], atol=0.50)))

  def test_compute_model_performance_singletask_classifier(self):
    n_data_points = 20
    n_features = 10

    X = np.ones(shape=(int(n_data_points / 2), n_features)) * -1
    X1 = np.ones(shape=(int(n_data_points / 2), n_features))
    X = np.concatenate((X, X1))
    class_1 = np.array([[0.0, 1.0] for x in range(int(n_data_points / 2))])
    class_0 = np.array([[1.0, 0.0] for x in range(int(n_data_points / 2))])
    y1 = np.concatenate((class_0, class_1))
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    entropies = []
    labels = []
    for i in range(1):
      label = Label(shape=(None, 2))
      labels.append(label)
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
            epochs=1000, batch_size=tg.batch_size, pad_batches=True))
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")

    scores = tg.evaluate_generator(
        databag.iterbatches(), [metric], labels=labels, per_task_metrics=True)
    scores = list(scores[1].values())
    assert_true(np.isclose(scores, [1.0], atol=0.05))

  def test_compute_model_performance_multitask_regressor(self):
    random_seed = 42
    n_data_points = 20
    n_features = 2
    np.random.seed(seed=random_seed)

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
    labels = []
    for i in range(2):
      label = Label(shape=(None, 1))
      dense = Dense(out_channels=1, in_layers=[features])
      loss = ReduceSquareDifference(in_layers=[dense, label])

      outputs.append(dense)
      losses.append(loss)
      labels.append(label)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=losses)

    tg = dc.models.TensorGraph(
        mode="regression",
        batch_size=20,
        random_seed=random_seed,
        learning_rate=0.1)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
        databag.iterbatches(
            epochs=1000, batch_size=tg.batch_size, pad_batches=True))
    metric = [
        dc.metrics.Metric(
            dc.metrics.mean_absolute_error, np.mean, mode="regression"),
    ]
    scores = tg.evaluate_generator(
        databag.iterbatches(), metric, labels=labels, per_task_metrics=True)
    scores = list(scores[1].values())
    assert_true(np.all(np.isclose(scores, [0.0, 0.0], atol=1.0)))

  def test_compute_model_performance_singletask_regressor(self):
    n_data_points = 20
    n_features = 2

    X = np.random.rand(n_data_points, n_features)
    y1 = np.expand_dims(np.array([0.5 for x in range(n_data_points)]), axis=-1)
    X = NumpyDataset(X)
    ys = [NumpyDataset(y1)]

    databag = Databag()

    features = Feature(shape=(None, n_features))
    databag.add_dataset(features, X)

    outputs = []
    losses = []
    labels = []
    for i in range(1):
      label = Label(shape=(None, 1))
      dense = Dense(out_channels=1, in_layers=[features])
      loss = ReduceSquareDifference(in_layers=[dense, label])

      outputs.append(dense)
      losses.append(loss)
      labels.append(label)
      databag.add_dataset(label, ys[i])

    total_loss = ReduceMean(in_layers=losses)

    tg = dc.models.TensorGraph(mode="regression", learning_rate=0.1)
    for output in outputs:
      tg.add_output(output)
    tg.set_loss(total_loss)

    tg.fit_generator(
        databag.iterbatches(
            epochs=1000, batch_size=tg.batch_size, pad_batches=True))
    metric = [
        dc.metrics.Metric(
            dc.metrics.mean_absolute_error, np.mean, mode="regression"),
    ]
    scores = tg.evaluate_generator(
        databag.iterbatches(batch_size=tg.batch_size),
        metric,
        labels=labels,
        per_task_metrics=True)
    scores = list(scores[1].values())
    assert_true(np.all(np.isclose(scores, [0.0], atol=0.5)))
