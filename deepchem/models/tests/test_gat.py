import unittest

from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models import GATModel
from deepchem.models.tests.test_graph_models import get_dataset

try:
  import torch  # noqa
  import torch_geometric  # noqa
  has_pytorch_and_pyg = True
except:
  has_pytorch_and_pyg = False


@unittest.skipIf(not has_pytorch_and_pyg,
                 'PyTorch and PyTorch Geometric are not installed')
def test_gat_regression():
  # load datasets
  featurizer = MolGraphConvFeaturizer()
  tasks, dataset, transformers, metric = get_dataset(
      'regression', featurizer=featurizer)

  # initialize models
  n_tasks = len(tasks)
  model = GATModel(mode='regression', n_tasks=n_tasks, batch_size=10)

  # overfit test
  # GAT's convergence is a little slow
  model.fit(dataset, nb_epoch=300)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean_absolute_error'] < 0.5


@unittest.skipIf(not has_pytorch_and_pyg,
                 'PyTorch and PyTorch Geometric are not installed')
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
      batch_size=10,
      learning_rate=0.001)

  # overfit test
  # GAT's convergence is a little slow
  model.fit(dataset, nb_epoch=150)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean-roc_auc_score'] >= 0.9
