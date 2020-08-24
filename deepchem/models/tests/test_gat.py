import unittest

from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models import GATModel, losses
from deepchem.models.tests.test_graph_models import get_dataset

try:
  import torch  # noqa
  import torch_geometric  # noqa
  has_pytorch_and_pyg = True
except:
  has_pytorch_and_pyg = False


@unittest.skipIf(not has_pytorch_and_pyg,
                 'PyTorch and PyTorch Geometric are not installed')
def test_gat_classification():
  # load datasets
  featurizer = MolGraphConvFeaturizer()
  tasks, dataset, transformers, metric = get_dataset(
      'regression', featurizer=featurizer)
  n_tasks = len(tasks)

  # initialize models
  model = GATModel(
      in_node_dim=25,
      hidden_node_dim=64,
      heads=1,
      num_conv=3,
      predicator_hidden_feats=32,
      n_tasks=n_tasks,
      loss=losses.L2Loss(),
      batch_size=10,
      learning_rate=0.001)

  # overfit test
  model.fit(dataset, nb_epoch=100)
  scores = model.evaluate(dataset, [metric], transformers)
  assert scores['mean_absolute_error'] < 0.1
