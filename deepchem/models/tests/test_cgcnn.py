import unittest
from os import path, remove

from deepchem.feat import CGCNNFeaturizer
from deepchem.molnet import load_perovskite
from deepchem.metrics import Metric, mae_score
from deepchem.models import CGCNNModel, losses

try:
  import dgl  # noqa
  import torch  # noqa
  has_pytorch_and_dgl = True
except:
  has_pytorch_and_dgl = False


@unittest.skipIf(not has_pytorch_and_dgl, 'PyTorch and DGL are not installed')
def test_cgcnn():
  # load datasets
  current_dir = path.dirname(path.abspath(__file__))
  config = {
      "reload": False,
      "featurizer": CGCNNFeaturizer,
      # disable transformer
      "transformers": [],
      # load 'deepchem/models/test/perovskite.tar.gz'
      "data_dir": current_dir
  }
  tasks, datasets, transformers = load_perovskite(**config)
  train, valid, test = datasets

  # initialize models
  n_tasks = 1
  model = CGCNNModel(
      in_node_dim=92,
      hidden_node_dim=64,
      in_edge_dim=41,
      num_conv=3,
      predicator_hidden_feats=128,
      n_tasks=n_tasks,
      loss=losses.L2Loss(),
      batch_size=32,
      learning_rate=0.001)

  # check train
  model.fit(train, nb_epoch=50)

  # check predict
  valid_preds = model.predict_on_batch(valid.X)
  assert valid_preds.shape == (10, n_tasks)
  test_preds = model.predict(test)
  assert test_preds.shape == (10, n_tasks)

  # eval model on test
  regression_metric = Metric(mae_score, n_tasks=n_tasks)
  scores = model.evaluate(test, [regression_metric])
  assert scores[regression_metric.name] < 1.0

  if path.exists(path.join(current_dir, 'perovskite.json')):
    remove(path.join(current_dir, 'perovskite.json'))
