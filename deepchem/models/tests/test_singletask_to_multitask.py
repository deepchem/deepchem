"""
Testing singletask-to-multitask.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import numpy as np
from deepchem.models.tests import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.datasets import DiskDataset
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.models.multitask import SingletaskToMultitask 
from deepchem.models.sklearn_models import SklearnModel
from sklearn.linear_model import LogisticRegression
from deepchem.utils.evaluate import Evaluator

class TestSingletasktoMultitaskAPI(TestAPI):
  """
  Test top-level API for singletask_to_multitask ML models.
  """
  def test_singletask_to_multitask_classification(self):
    n_features = 10
    n_tasks = 17
    tasks = range(n_tasks)
    # Define train dataset
    n_train = 100
    X_train = np.random.rand(n_train, n_features)
    y_train = np.random.randint(2, size=(n_train, n_tasks))
    w_train = np.ones_like(y_train)
    ids_train = ["C"] * n_train
    train_dataset = DiskDataset.from_numpy(
        self.train_dir, X_train, y_train, w_train, ids_train)

    # Define test dataset
    n_test = 10
    X_test = np.random.rand(n_test, n_features)
    y_test = np.random.randint(2, size=(n_test, n_tasks))
    w_test = np.ones_like(y_test)
    ids_test = ["C"] * n_test
    test_dataset = DiskDataset.from_numpy(
        self.test_dir, X_test, y_test, w_test, ids_test)

    transformers = []
    classification_metrics = [Metric(metrics.roc_auc_score)]
    def model_builder(model_dir):
      sklearn_model = LogisticRegression()
      return SklearnModel(sklearn_model, model_dir)
    multitask_model = SingletaskToMultitask(
        tasks, model_builder, self.model_dir)

    # Fit trained model
    multitask_model.fit(train_dataset)
    multitask_model.save()

    # Eval multitask_model on train
    evaluator = Evaluator(multitask_model, train_dataset, transformers,
                          verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)

    # Eval multitask_model on test
    evaluator = Evaluator(multitask_model, test_dataset, transformers,
                          verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)
