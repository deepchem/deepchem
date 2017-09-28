import numpy as np

import deepchem
from deepchem.data import NumpyDataset
from deepchem.models import GraphConvTensorGraph
from deepchem.models import TensorGraph
from deepchem.molnet.load_function.delaney_datasets import load_delaney


def get_dataset(mode='classification', featurizer='GraphConv'):
  data_points = 10
  tasks, all_dataset, transformers = load_delaney(featurizer)
  train, valid, test = all_dataset

  if mode == 'classification':
    y = np.random.randint(0, 2, size=(data_points, len(tasks)))
    metric = deepchem.metrics.Metric(
        deepchem.metrics.roc_auc_score, np.mean, mode="classification")
  if mode == 'regression':
    y = np.random.normal(size=(data_points, len(tasks)))
    metric = deepchem.metrics.Metric(
        deepchem.metrics.mean_absolute_error, mode="regression")

  ds = NumpyDataset(train.X[:10], y, train.w[:10], train.ids[:10])

  return tasks, ds, transformers, metric


def test_graph_conv_model():
  tasks, dataset, transformers, metric = get_dataset('classification',
                                                     'GraphConv')

  batch_size = 50
  model = GraphConvTensorGraph(
      len(tasks), batch_size=batch_size, mode='classification')

  model.fit(dataset, nb_epoch=1)
  scores = model.evaluate(dataset, [metric], transformers)

  model.save()
  model = TensorGraph.load_from_dir(model.model_dir)
  scores = model.evaluate(dataset, [metric], transformers)


def test_graph_conv_regression_model():
  tasks, dataset, transformers, metric = get_dataset('regression', 'GraphConv')

  batch_size = 50
  model = GraphConvTensorGraph(
      len(tasks), batch_size=batch_size, mode='regression')

  model.fit(dataset, nb_epoch=1)
  scores = model.evaluate(dataset, [metric], transformers)

  model.save()
  model = TensorGraph.load_from_dir(model.model_dir)
  scores = model.evaluate(dataset, [metric], transformers)


def test_graph_conv_error_bars():
  tasks, dataset, transformers, metric = get_dataset('regression', 'GraphConv')

  batch_size = 50
  model = GraphConvTensorGraph(
      len(tasks), batch_size=batch_size, mode='regression')

  model.fit(dataset, nb_epoch=1)

  mu, sigma = model.bayesian_predict(
      dataset, transformers, untransform=True, n_passes=24)
  assert mu.shape == (len(dataset), len(tasks))
  assert sigma.shape == (len(dataset), len(tasks))
