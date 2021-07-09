import unittest
import tempfile

import numpy as np

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models.tests.test_graph_models import get_dataset

try:
  import dgl
  import dgllife
  import torch
  from deepchem.models.torch_models import MPNNModel
  has_torch_and_dgl = True
except:
  has_torch_and_dgl = False


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_mpnn_regression():
  # load datasets
  featurizer = MolGraphConvFeaturizer(use_edges=True)
  tasks, dataset, transformers, metric = get_dataset(
      'regression', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model = MPNNModel(mode='regression', n_tasks=n_tasks, learning_rate=0.0005)

  # overfit test
  model.fit(dataset, nb_epoch=400)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean_absolute_error'] < 0.5

  # test on a small MoleculeNet dataset
  from deepchem.molnet import load_delaney

  tasks, all_dataset, transformers = load_delaney(featurizer=featurizer)
  train_set, _, _ = all_dataset
  model = MPNNModel(
      mode='regression',
      n_tasks=len(tasks),
      node_out_feats=2,
      edge_hidden_feats=2,
      num_step_message_passing=1,
      num_step_set2set=1,
      num_layer_set2set=1)
  model.fit(train_set, nb_epoch=1)


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_mpnn_classification():
  # load datasets
  featurizer = MolGraphConvFeaturizer(use_edges=True)
  tasks, dataset, transformers, metric = get_dataset(
      'classification', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model = MPNNModel(
      mode='classification', n_tasks=n_tasks, learning_rate=0.0005)

  # overfit test
  model.fit(dataset, nb_epoch=200)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.8

  # test on a small MoleculeNet dataset
  from deepchem.molnet import load_bace_classification

  tasks, all_dataset, transformers = load_bace_classification(
      featurizer=featurizer)
  train_set, _, _ = all_dataset
  model = MPNNModel(
      mode='classification',
      n_tasks=len(tasks),
      node_out_feats=2,
      edge_hidden_feats=2,
      num_step_message_passing=1,
      num_step_set2set=1,
      num_layer_set2set=1)
  model.fit(train_set, nb_epoch=1)


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_mpnn_reload():
  # load datasets
  featurizer = MolGraphConvFeaturizer(use_edges=True)
  tasks, dataset, transformers, metric = get_dataset(
      'classification', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model_dir = tempfile.mkdtemp()
  model = MPNNModel(
      mode='classification',
      n_tasks=n_tasks,
      model_dir=model_dir,
      batch_size=10,
      learning_rate=0.001)

  model.fit(dataset, nb_epoch=200)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.85

  reloaded_model = MPNNModel(
      mode='classification',
      n_tasks=n_tasks,
      model_dir=model_dir,
      batch_size=10,
      learning_rate=0.001)
  reloaded_model.restore()

  pred_mols = ["CCCC", "CCCCCO", "CCCCC"]
  X_pred = featurizer(pred_mols)
  random_dataset = dc.data.NumpyDataset(X_pred)
  original_pred = model.predict(random_dataset)
  reload_pred = reloaded_model.predict(random_dataset)
  assert np.all(original_pred == reload_pred)
