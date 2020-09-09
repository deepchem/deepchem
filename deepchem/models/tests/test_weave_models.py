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


def get_dataset(mode='classification',
                featurizer='GraphConv',
                num_tasks=2,
                data_points=20):
  if mode == 'classification':
    tasks, all_dataset, transformers = load_bace_classification(
        featurizer, reload=False)
  else:
    tasks, all_dataset, transformers = load_delaney(featurizer, reload=False)

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


def test_compute_features_on_infinity_distance():
  """Test that WeaveModel correctly transforms WeaveMol objects into tensors with infinite max_pair_distance."""
  featurizer = dc.feat.WeaveFeaturizer(max_pair_distance=None)
  X = featurizer(["C", "CCC"])
  batch_size = 20
  model = WeaveModel(
      1,
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
  atom_feat, pair_feat, pair_split, atom_split, atom_to_pair = model.compute_features_on_batch(
      X)

  # There are 4 atoms each of which have 75 atom features
  assert atom_feat.shape == (4, 75)
  # There are 10 pairs with infinity distance and 14 pair features
  assert pair_feat.shape == (10, 14)
  # 4 atoms in total
  assert atom_split.shape == (4,)
  assert np.all(atom_split == np.array([0, 1, 1, 1]))
  # 10 pairs in total
  assert pair_split.shape == (10,)
  assert np.all(pair_split == np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
  # 10 pairs in total each with start/finish
  assert atom_to_pair.shape == (10, 2)
  assert np.all(
      atom_to_pair == np.array([[0, 0], [1, 1], [1, 2], [1, 3], [2, 1], [2, 2],
                                [2, 3], [3, 1], [3, 2], [3, 3]]))


def test_compute_features_on_distance_1():
  """Test that WeaveModel correctly transforms WeaveMol objects into tensors with finite max_pair_distance."""
  featurizer = dc.feat.WeaveFeaturizer(max_pair_distance=1)
  X = featurizer(["C", "CCC"])
  batch_size = 20
  model = WeaveModel(
      1,
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
  atom_feat, pair_feat, pair_split, atom_split, atom_to_pair = model.compute_features_on_batch(
      X)

  # There are 4 atoms each of which have 75 atom features
  assert atom_feat.shape == (4, 75)
  # There are 8 pairs with distance 1 and 14 pair features. (To see why 8,
  # there's the self pair for "C". For "CCC" there are 7 pairs including self
  # connections and accounting for symmetry.)
  assert pair_feat.shape == (8, 14)
  # 4 atoms in total
  assert atom_split.shape == (4,)
  assert np.all(atom_split == np.array([0, 1, 1, 1]))
  # 10 pairs in total
  assert pair_split.shape == (8,)
  # The center atom is self connected and to both neighbors so it appears
  # thrice. The canonical ranking used in MolecularFeaturizer means this
  # central atom is ranked last in ordering.
  assert np.all(pair_split == np.array([0, 1, 1, 2, 2, 3, 3, 3]))
  # 10 pairs in total each with start/finish
  assert atom_to_pair.shape == (8, 2)
  assert np.all(atom_to_pair == np.array([[0, 0], [1, 1], [1, 3], [2, 2],
                                          [3, 3], [3, 1], [3, 2], [3, 3]]))


@flaky
@pytest.mark.slow
def test_weave_model():
  tasks, dataset, transformers, metric = get_dataset(
      'classification', 'Weave', data_points=10)

  batch_size = 10
  model = WeaveModel(
      len(tasks),
      batch_size=batch_size,
      mode='classification',
      final_conv_activation_fn=None,
      dropouts=0,
      learning_rage=0.0003)
  model.fit(dataset, nb_epoch=100)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.slow
def test_weave_regression_model():
  import numpy as np
  import tensorflow as tf
  tf.random.set_seed(123)
  np.random.seed(123)
  tasks, dataset, transformers, metric = get_dataset(
      'regression', 'Weave', data_points=10)

  batch_size = 10
  model = WeaveModel(
      len(tasks),
      batch_size=batch_size,
      mode='regression',
      dropouts=0,
      learning_rate=0.00003)
  model.fit(dataset, nb_epoch=400)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean_absolute_error'] < 0.1


def test_weave_fit_simple_infinity_distance():
  featurizer = dc.feat.WeaveFeaturizer(max_pair_distance=None)
  X = featurizer(["C", "CCC"])
  y = np.array([0, 1.])
  dataset = dc.data.NumpyDataset(X, y)

  batch_size = 20
  model = WeaveModel(
      1,
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
  transformers = []
  metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.9


def test_weave_fit_simple_distance_1():
  featurizer = dc.feat.WeaveFeaturizer(max_pair_distance=1)
  X = featurizer(["C", "CCC"])
  y = np.array([0, 1.])
  dataset = dc.data.NumpyDataset(X, y)

  batch_size = 20
  model = WeaveModel(
      1,
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
  transformers = []
  metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.9
