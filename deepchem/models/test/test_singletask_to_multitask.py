"""
Testing singletask-to-multitask.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

from deepchem.models.test import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
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
    output_transformer_classes = []
    input_transformer_classes = []
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: "classification" for task in tasks}
    input_file = "multitask_example.csv"
    train_dataset, test_dataset, _, output_transformers, = \
        self._featurize_train_test_split(
            splittype, compound_featurizers, 
            complex_featurizers, input_transformer_classes,
            output_transformer_classes, input_file, task_types.keys())
    params_dict = {
        "batch_size": 32,
        "data_shape": train_dataset.get_data_shape()
    }
    classification_metrics = [Metric(metrics.roc_auc_score)]
    def model_builder(task_types, model_params, verbosity=None):
      return SklearnModel(task_types, model_params,
                          model_instance=LogisticRegression())
    multitask_model = SingletaskToMultitask(task_types, params_dict, model_builder)
    self._create_model(train_dataset, test_dataset, multitask_model,
                       output_transformers, classification_metrics)
