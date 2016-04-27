"""
Tests to make sure TF models can overfit on tiny datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import tempfile
import numpy as np
import unittest
from deepchem import metrics
from deepchem.datasets import Dataset
from deepchem.metrics import Metric
from deepchem.models.test import TestAPI
from deepchem.utils.evaluate import Evaluator
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier

class TestTensorflowAPI(TestAPI):
  """
  Test top-level API for ML models."
  """

  def test_classification_overfit(self):
    """Test that data associated with a tasks stays associated with it."""
    tasks = ["task0"]
    task_types = {task: "classification" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    
    # Generate dummy dataset
    ids = np.arange(n_samples)
    #X = np.random.rand(n_samples, n_features)
    X = np.ones((n_samples, n_features))
    #y = np.random.randint(2, size=(n_samples, n_tasks))
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, tasks, X, y, w, ids)

    model_params = {
      "batch_size": 2,
      "num_classification_tasks": 1,
      "num_features": n_features,
      "layer_sizes": [1024],
      "weight_init_stddevs": [.01],
      "bias_init_consts": [0.],
      "dropouts": [.5],
      "num_classes": 2,
      "nb_epoch": 100,
      "penalty": 0.0,
      "optimizer": "sgd",
      "learning_rate": .0003,
      "data_shape": dataset.get_data_shape()
    }

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score)
    model = TensorflowModel(
        tasks, task_types, model_params, self.model_dir,
        tf_class=TensorflowMultiTaskClassifier,
        verbosity=verbosity)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    with tempfile.NamedTemporaryFile() as csv_out:
      with tempfile.NamedTemporaryFile() as stats_out:
        multitask_scores = evaluator.compute_model_performance(
            [classification_metric], csv_out.name, stats_out)

    print("multitask_scores")
    print(multitask_scores)
    assert multitask_scores[classification_metric.name] > .9
