import unittest
import tempfile

import numpy as np

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models import GATModel
from deepchem.models.tests.test_graph_models import get_dataset

try:
  import dgl
  import dgllife
  import torch
  has_torch_and_dgl = True
except:
  has_torch_and_dgl = False


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_gat_regression():
  # load datasets
  featurizer = MolGraphConvFeaturizer()
  tasks, dataset, transformers, metric = get_dataset(
      'regression', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model = GATModel(
      mode='regression',
      n_tasks=n_tasks,
      number_atom_features=30,
      batch_size=10)

  # overfit test
  model.fit(dataset, nb_epoch=150)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean_absolute_error'] < 0.5


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_gat_classification():
  # load datasets
  featurizer = MolGraphConvFeaturizer()
  tasks, dataset, transformers, metric = get_dataset(
      'classification', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model = GATModel(
      mode='classification',
      n_tasks=n_tasks,
      number_atom_features=30,
      batch_size=10,
      learning_rate=0.001)

  # overfit test
  model.fit(dataset, nb_epoch=80)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.85


@unittest.skipIf(not has_torch_and_dgl,
                 'PyTorch, DGL, or DGL-LifeSci are not installed')
def test_gat_reload():
  # load datasets
  featurizer = MolGraphConvFeaturizer()
  tasks, dataset, transformers, metric = get_dataset(
      'classification', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model_dir = tempfile.mkdtemp()
  model = GATModel(
      mode='classification',
      n_tasks=n_tasks,
      number_atom_features=30,
      model_dir=model_dir,
      batch_size=10,
      learning_rate=0.001)

  model.fit(dataset, nb_epoch=60)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.85

  reloaded_model = GATModel(
      mode='classification',
      n_tasks=n_tasks,
      number_atom_features=30,
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
