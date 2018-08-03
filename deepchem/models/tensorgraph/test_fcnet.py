from unittest import TestCase

import deepchem as dc
import numpy as np
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.data import NumpyDataset
from deepchem.models import MultitaskRegressor
from deepchem.models import MultitaskClassifier


class TestMultitaskDense(TestCase):

  def get_dataset(self, mode='classification', featurizer='ECFP', num_tasks=2):
    data_points = 10
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

  def test_multitaskregressor_save_load(self):
    tasks, dataset, transformers, metric = self.get_dataset('regression')

    model = MultitaskRegressor(len(tasks), n_features=dataset.X.shape[1])
    model.fit(dataset, nb_epoch=1)
    scores = model.predict(dataset)
    model.save()

    model = MultitaskRegressor.load_from_dir(model.model_dir)
    scores2 = model.predict(dataset)
    self.assertTrue(np.all(scores == scores2))

  def test_multitaskclassifier_save_load(self):
    tasks, dataset, transformers, metric = self.get_dataset('classification')

    model = MultitaskClassifier(len(tasks), n_features=dataset.X.shape[1])
    model.fit(dataset, nb_epoch=1)
    scores = model.predict(dataset)
    model.save()

    model = MultitaskClassifier.load_from_dir(model.model_dir)
    scores2 = model.predict(dataset)
    self.assertTrue(np.all(scores == scores2))
