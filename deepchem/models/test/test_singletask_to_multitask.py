"""
Testing singletask-to-multitask.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import numpy as np
from deepchem.models.test import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.datasets import Dataset
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.models.multitask import SingletaskToMultitask 
from deepchem.models.sklearn_models import SklearnModel
from sklearn.linear_model import LogisticRegression

class TestSingletasktoMultitaskAPI(TestAPI):
  """
  Test top-level API for singletask_to_multitask ML models.
  """
  def test_singletask_to_multitask_classification(self):
    splittype = "scaffold"
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    output_transformers = []
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: "classification" for task in tasks}
    input_file = "multitask_example.csv"

    n_features = 10
    n_tasks = len(tasks)
    # Define train dataset
    n_train = 100
    X_train = np.random.rand(n_train, n_features)
    y_train = np.random.randint(2, size=(n_train, n_tasks))
    w_train = np.ones_like(y_train)
    ids_train = ["C"] * n_train
    train_dataset = Dataset.from_numpy(self.train_dir, tasks,
                                       X_train, y_train, w_train, ids_train)

    # Define test dataset
    n_test = 10
    X_test = np.random.rand(n_test, n_features)
    y_test = np.random.randint(2, size=(n_test, n_tasks))
    w_test = np.ones_like(y_test)
    ids_test = ["C"] * n_test
    test_dataset = Dataset.from_numpy(self.test_dir, tasks,
                                       X_test, y_test, w_test, ids_test)

    params_dict = {
        "batch_size": 32,
        "data_shape": train_dataset.get_data_shape()
    }
    classification_metrics = [Metric(metrics.roc_auc_score)]
    def model_builder(tasks, task_types, model_params, model_builder, verbosity=None):
      return SklearnModel(tasks, task_types, model_params, model_builder,
                          model_instance=LogisticRegression())
    multitask_model = SingletaskToMultitask(tasks, task_types, params_dict,
                                            self.model_dir, model_builder)
    self._create_model(train_dataset, test_dataset, multitask_model,
                       output_transformers, classification_metrics)
