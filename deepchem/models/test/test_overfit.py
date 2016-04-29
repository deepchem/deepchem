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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from deepchem import metrics
from deepchem.datasets import Dataset
from deepchem.metrics import Metric
from deepchem.models.test import TestAPI
from deepchem.utils.evaluate import Evaluator
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor

class TestOverfitAPI(TestAPI):
  """
  Test that sklearn and keras models can overfit simple datasets.
  """

  def test_sklearn_regression_overfit(self):
    """Test that sklearn models can overfit simple regression datasets."""
    tasks = ["task0"]
    task_types = {task: "regression" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks)
    w = np.ones((n_samples, n_tasks))

    dataset = Dataset.from_numpy(self.train_dir, tasks, X, y, w, ids)

    model_params = {
      "batch_size": None,
      "data_shape": dataset.get_data_shape()
    }

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="regression",
                         model_instance=RandomForestRegressor())

    # Fit trained model
    model.fit(dataset)
    model.save()

    y_pred_model = model.predict(dataset, transformers=[])

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    with tempfile.NamedTemporaryFile() as csv_out:
      with tempfile.NamedTemporaryFile() as stats_out:
        scores = evaluator.compute_model_performance(
            [regression_metric], csv_out.name, stats_out, proba=True)

    assert scores[regression_metric.name] > .7

  def test_sklearn_classification_overfit(self):
    """Test that sklearn models can overfit simple classification datasets."""
    tasks = ["task0"]
    task_types = {task: "classification" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, tasks, X, y, w, ids)

    model_params = {
      "batch_size": None,
      "data_shape": dataset.get_data_shape()
    }

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="classification",
                         model_instance=RandomForestClassifier())

    # Fit trained model
    model.fit(dataset)
    model.save()

    y_pred_model = model.predict(dataset, transformers=[])
    y_pred_proba_model = model.predict_proba(dataset, transformers=[])

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    with tempfile.NamedTemporaryFile() as csv_out:
      with tempfile.NamedTemporaryFile() as stats_out:
        scores = evaluator.compute_model_performance(
            [classification_metric], csv_out.name, stats_out, proba=True)

    print("scores")
    print(scores)

    assert scores[classification_metric.name] > .9

  def test_keras_regression_overfit(self):
    """Test that keras models can overfit simple regression datasets."""
    tasks = ["task0"]
    task_types = {task: "regression" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks)
    w = np.ones((n_samples, n_tasks))

    dataset = Dataset.from_numpy(self.train_dir, tasks, X, y, w, ids)

    model_params = {
        "nb_hidden": 1000,
        "activation": "relu",
        "dropout": .0,
        "learning_rate": .15,
        "momentum": .9,
        "nesterov": False,
        "decay": 1e-4,
        "batch_size": n_samples,
        "nb_epoch": 200,
        "init": "glorot_uniform",
        "nb_layers": 1,
        "batchnorm": False,
        "data_shape": dataset.get_data_shape()
    }

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    model = MultiTaskDNN(tasks, task_types, model_params, self.model_dir,
                         verbosity=verbosity)

    # Fit trained model
    model.fit(dataset)
    model.save()

    y_pred_model = model.predict(dataset, transformers=[])

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    with tempfile.NamedTemporaryFile() as csv_out:
      with tempfile.NamedTemporaryFile() as stats_out:
        scores = evaluator.compute_model_performance(
            [regression_metric], csv_out.name, stats_out, proba=True)
    print("scores")
    print(scores)

    assert scores[regression_metric.name] > .7

  def test_tf_regression_overfit(self):
    """Test that TensorFlow models can overfit simple regression datasets."""
    tasks = ["task0"]
    task_types = {task: "regression" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks)
    w = np.ones((n_samples, n_tasks))

    dataset = Dataset.from_numpy(self.train_dir, tasks, X, y, w, ids)

    model_params = {
      "batch_size": n_samples,
      "num_regression_tasks": 1,
      "num_features": n_features,
      "layer_sizes": [1000],
      "weight_init_stddevs": [.01],
      "bias_init_consts": [0.5],
      "dropouts": [.0],
      "num_classes": 2,
      "nb_epoch": 100,
      "penalty": 0.0,
      "optimizer": "sgd",
      "learning_rate": .0001,
      "data_shape": dataset.get_data_shape()
    }

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    model = TensorflowModel(
        tasks, task_types, model_params, self.model_dir,
        tf_class=TensorflowMultiTaskRegressor,
        verbosity=verbosity)

    # Fit trained model
    model.fit(dataset)
    model.save()

    y_pred_model = model.predict(dataset, transformers=[])

    # Eval model on train
    transformers = []
    evaluator = Evaluator(model, dataset, transformers, verbosity=verbosity)
    with tempfile.NamedTemporaryFile() as csv_out:
      with tempfile.NamedTemporaryFile() as stats_out:
        scores = evaluator.compute_model_performance(
            [regression_metric], csv_out.name, stats_out, proba=True)
    print("scores")
    print(scores)

    assert scores[regression_metric.name] > .7
