import unittest
import os
import numpy as np
import pytest
import scipy

import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.models import GraphConvModel, DAGModel, WeaveModel, MPNNModel
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.feat import ConvMolFeaturizer

from flaky import flaky


def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
  data_points = 20
  if mode == 'classification':
    tasks, all_dataset, transformers = load_bace_classification(featurizer)
  else:
    tasks, all_dataset, transformers = load_delaney(featurizer)

  train, valid, test = all_dataset
  for i in range(1, num_tasks):
    tasks.append("random_task")
  w = np.ones(shape=(data_points, len(tasks)))

  if mode == 'classification':
    y = np.random.randint(0, 2, size=(data_points, len(tasks)))
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
  else:
    y = np.random.normal(size=(data_points, len(tasks)))
    metric = dc.metrics.Metric(
        dc.metrics.mean_absolute_error, mode="regression")

  ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

  return tasks, ds, transformers, metric


@flaky
@pytest.mark.slow
def test_weave_model():
  tasks, dataset, transformers, metric = get_dataset('classification', 'Weave')

  batch_size = 20
  model = WeaveModel(
      len(tasks),
      batch_size=batch_size,
      mode='classification',
      fully_connected_layer_sizes=[2000, 1000],
      batch_normalize=True,
      batch_normalize_kwargs={
          "fused": False,
          "trainable": True,
          "renorm": True
      },
      learning_rage=0.0005)
  model.fit(dataset, nb_epoch=200)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.slow
def test_weave_regression_model():
  import numpy as np
  import tensorflow as tf
  tf.random.set_seed(123)
  np.random.seed(123)
  tasks, dataset, transformers, metric = get_dataset('regression', 'Weave')

  batch_size = 10
  model = WeaveModel(
      len(tasks),
      batch_size=batch_size,
      mode='regression',
      batch_normalize=False,
      fully_connected_layer_sizes=[],
      dropouts=0,
      learning_rate=0.0005)
  model.fit(dataset, nb_epoch=200)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean_absolute_error'] < 0.1


def test_weave_fit_simple():
  featurizer = dc.feat.WeaveFeaturizer()
  X = featurizer(["C", "CCC"])
  y = np.random.randint(2, size=(2,))
  dataset = dc.data.NumpyDataset(X, y)
  tasks, dataset, transformers, metric = get_dataset('classification', 'Weave')

  batch_size = 20
  model = WeaveModel(
      len(tasks),
      batch_size=batch_size,
      mode='classification',
      fully_connected_layer_sizes=[2000, 1000],
      batch_normalize=True,
      batch_normalize_kwargs={
          "fused": False,
          "trainable": True,
          "renorm": True
      },
      learning_rage=0.0005)
  model.fit(dataset, nb_epoch=200)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.9
