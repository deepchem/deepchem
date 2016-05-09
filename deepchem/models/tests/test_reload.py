"""
Test reload for trained models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.tests import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.datasets import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier

class TestModelReload(TestAPI):

  def test_sklearn_reload(self):
    """Test that trained model can be reloaded correctly."""
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

    # Load trained model
    reloaded_model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                                  mode="classification")
    reloaded_model.reload()

    # Eval model on train
    transformers = []
    evaluator = Evaluator(reloaded_model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_keras_reload(self):
    """Test that trained keras models can be reloaded correctly."""
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
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    model = MultiTaskDNN(tasks, task_types, model_params, self.model_dir,
                         verbosity=verbosity)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Load trained model
    reloaded_model = MultiTaskDNN(tasks, task_types, model_params, self.model_dir,
                                  verbosity=verbosity)
    reloaded_model.reload()
    

    # Eval model on train
    transformers = []
    evaluator = Evaluator(reloaded_model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9

  def test_tf_reload(self):
    """Test that tensorflow models can overfit simple classification datasets."""
    tasks = ["task0"]
    task_types = {task: "classification" for task in tasks}
    n_samples = 10
    n_features = 3
    n_tasks = len(tasks)
    n_classes = 2
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
  
    dataset = Dataset.from_numpy(self.train_dir, tasks, X, y, w, ids)

    model_params = {
      "layer_sizes": [1000],
      "dropouts": [0.],
      "learning_rate": 0.003,
      "momentum": .9,
      "batch_size": n_samples,
      "num_classification_tasks": 1,
      "num_classes": n_classes,
      "num_features": n_features,
      "weight_init_stddevs": [1.],
      "bias_init_consts": [1.],
      "nb_epoch": 100,
      "penalty": 0.0,
      "optimizer": "adam",
      "data_shape": dataset.get_data_shape()
    }

    verbosity = "high"
    classification_metric = Metric(metrics.accuracy_score, verbosity=verbosity)
    model = TensorflowModel(
        tasks, task_types, model_params, self.model_dir,
        tf_class=TensorflowMultiTaskClassifier,
        verbosity=verbosity)

    # Fit trained model
    model.fit(dataset)
    model.save()

    # Load trained model
    reloaded_model = TensorflowModel(
        tasks, task_types, model_params, self.model_dir,
        tf_class=TensorflowMultiTaskClassifier,
        verbosity=verbosity)
    reloaded_model.reload()
    assert reloaded_model.eval_model._restored_model

    # Eval model on train
    transformers = []
    evaluator = Evaluator(reloaded_model, dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])

    assert scores[classification_metric.name] > .9
