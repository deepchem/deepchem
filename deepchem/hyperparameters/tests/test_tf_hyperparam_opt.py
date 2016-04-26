"""
Integration tests for hyperparam optimization.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"


import os
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.models.test import TestAPI
from deepchem.metrics import Metric
from deepchem import metrics
import numpy as np
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier

class TestTFHyperparamOptAPI(TestAPI):
  """
  Test hyperparameter optimization API.
  """
  def test_multitask_tf_mlp_ECFP_classification_hyperparam_opt(self):
    """Straightforward test of Tensorflow multitask deepchem classification API."""
    splittype = "scaffold"
    output_transformers = []
    input_transformers = []
    task_type = "classification"

    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: task_type for task in tasks}

    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []

    train_dataset, valid_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks)
    metric = Metric(metrics.matthews_corrcoef, np.mean)
    params_dict = {"activation": ["relu"],
                    "momentum": [.9],
                    "batch_size": [50],
                    "init": ["glorot_uniform"],
                    "data_shape": [train_dataset.get_data_shape()],
                    "learning_rate": [1e-3],
                    "decay": [1e-6],
                    "nb_hidden": [1000], 
                    "nb_epoch": [1],
                    "nesterov": [False],
                    "dropouts": [(.5,)],
                    "nb_layers": [1],
                    "batchnorm": [False],
                    "layer_sizes": [(1000,)],
                    "weight_init_stddevs": [(.1,)],
                    "bias_init_consts": [(1.,)],
                    "num_classes": [2],
                    "penalty": [0.], 
                    "optimizer": ["sgd"],
                    "num_classification_tasks": [len(task_types)]
                  }

    def model_builder(tasks, task_types, params_dict, logdir, verbosity=None):
        return TensorflowModel(
            tasks, task_types, params_dict, logdir, 
            tf_class=TensorflowMultiTaskClassifier,
            verbosity=verbosity)
    self._hyperparam_opt(model_builder, params_dict, train_dataset,
                         valid_dataset, output_transformers, tasks, task_types,
                         metric, logdir=self.model_dir)
