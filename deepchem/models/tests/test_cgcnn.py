import unittest
from os import path, remove

from deepchem.feat import CGCNNFeaturizer
from deepchem.molnet import load_perovskite, load_mp_metallicity
from deepchem.metrics import Metric, mae_score, roc_auc_score
from deepchem.models import CGCNNModel

try:
  import dgl  # noqa
  import torch  # noqa
  has_pytorch_and_dgl = True
except:
  has_pytorch_and_dgl = False


@unittest.skipIf(not has_pytorch_and_dgl, 'PyTorch and DGL are not installed')
def test_cgcnn():
  # regression test
  # load datasets
  current_dir = path.dirname(path.abspath(__file__))
  config = {
      "reload": False,
      "featurizer": CGCNNFeaturizer,
      # disable transformer
      "transformers": [],
      "data_dir": current_dir
  }
  tasks, datasets, transformers = load_perovskite(**config)
  train, valid, test = datasets

  n_tasks = len(tasks)
  model = CGCNNModel(
      n_tasks=n_tasks, mode='regression', batch_size=4, learning_rate=0.001)

  # check train
  model.fit(train, nb_epoch=20)

  # check predict shape
  valid_preds = model.predict_on_batch(valid.X)
  assert valid_preds.shape == (2, n_tasks)
  test_preds = model.predict(test)
  assert test_preds.shape == (3, n_tasks)

  # check overfit
  regression_metric = Metric(mae_score, n_tasks=n_tasks)
  scores = model.evaluate(train, [regression_metric], transformers)
  assert scores[regression_metric.name] < 0.6

  # classification test
  tasks, datasets, transformers = load_mp_metallicity(**config)
  train, valid, test = datasets

  # load datasets
  n_tasks = len(tasks)
  n_classes = 2
  model = CGCNNModel(
      n_tasks=n_tasks,
      n_classes=n_classes,
      mode='classification',
      batch_size=4,
      learning_rate=0.001)

  # check train
  model.fit(train, nb_epoch=20)

  # check predict shape
  valid_preds = model.predict_on_batch(valid.X)
  assert valid_preds.shape == (2, n_classes)
  test_preds = model.predict(test)
  assert test_preds.shape == (3, n_classes)

  # check overfit
  classification_metric = Metric(roc_auc_score, n_tasks=n_tasks)
  scores = model.evaluate(
      train, [classification_metric], transformers, n_classes=n_classes)
  assert scores[classification_metric.name] > 0.8

  # TODO: Multi task classification test

  if path.exists(path.join(current_dir, 'perovskite.json')):
    remove(path.join(current_dir, 'perovskite.json'))
  if path.exists(path.join(current_dir, 'mp_is_metal.json')):
    remove(path.join(current_dir, 'mp_is_metal.json'))
