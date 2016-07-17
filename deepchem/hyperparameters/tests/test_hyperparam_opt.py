"""
Integration tests for hyperparam optimization.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import unittest
import tempfile
import shutil
import numpy as np
from deepchem.models.tests import TestAPI
from deepchem.models.sklearn_models import SklearnModel
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.transformers import NormalizationTransformer
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.multitask import SingletaskToMultitask 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor 
from deepchem.datasets import Dataset
from deepchem.hyperparameters import HyperparamOpt
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.splits import ScaffoldSplitter

def rf_model_builder(tasks, task_types, params_dict, model_dir, verbosity=None):
    """Builds random forests given hyperparameters.

    Last two arguments only for tensorflow models and ignored.
    """
    n_estimators = params_dict["n_estimators"]
    max_features = params_dict["max_features"]
    return SklearnModel(
        tasks, task_types, params_dict, model_dir,
        mode="regression",
        model_instance=RandomForestRegressor(n_estimators=n_estimators,
                                             max_features=max_features))

class TestHyperparamOptAPI(TestAPI):
  """
  Test hyperparameter optimization API.
  """
  def test_singletask_sklearn_rf_ECFP_regression_hyperparam_opt(self):
    """Test of hyperparam_opt with singletask RF ECFP regression API."""
    splittype = "scaffold"
    featurizer = CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example.csv")
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                featurizer=featurizer,
                                verbosity="low")
    dataset = featurizer.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)

    input_transformers = []
    output_transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    transformers = input_transformers + output_transformers
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        transformer.transform(dataset)
    params_dict = {
      "n_estimators": [10, 100],
      "max_features": ["auto"],
      "data_shape": train_dataset.get_data_shape()
    }
    metric = Metric(metrics.r2_score)

    optimizer = HyperparamOpt(rf_model_builder, tasks, task_types, verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, output_transformers,
      metric, logdir=None)

  def test_singletask_to_multitask_sklearn_hyperparam_opt(self):
    """Test of hyperparam_opt with singletask_to_multitask."""
    splittype = "scaffold"
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
    train_dataset = Dataset.from_numpy(self.train_dir,
                                       X_train, y_train, w_train, ids_train,
                                       tasks)

    # Define validation dataset
    n_valid = 10
    X_valid = np.random.rand(n_valid, n_features)
    y_valid = np.random.randint(2, size=(n_valid, n_tasks))
    w_valid = np.ones_like(y_valid)
    ids_valid = ["C"] * n_valid
    valid_dataset = Dataset.from_numpy(self.valid_dir,
                                       X_valid, y_valid, w_valid, ids_valid,
                                       tasks)
    params_dict = {
        "batch_size": [32],
        "data_shape": [train_dataset.get_data_shape()],
    }
    classification_metric = Metric(metrics.matthews_corrcoef, np.mean,
                                   mode="classification")
    def model_builder(tasks, task_types, model_params, task_model_dir,
                      verbosity=None):
      return SklearnModel(tasks, task_types, model_params, task_model_dir,
                          model_instance=LogisticRegression())
    def multitask_model_builder(tasks, task_types, params_dict, logdir=None,
                                verbosity=None):
      return SingletaskToMultitask(tasks, task_types, params_dict,
                                   self.model_dir, model_builder)

    optimizer = HyperparamOpt(multitask_model_builder, tasks, task_types,
                              verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, output_transformers,
      classification_metric, logdir=None)

  def test_multitask_keras_mlp_ECFP_classification_hyperparam_opt(self):
    """Straightforward test of Keras multitask deepchem classification API."""
    task_type = "classification"
    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: task_type for task in tasks}

    featurizer = [CircularFingerprint(size=1024)]
    data_featurizer = DataFeaturizer(tasks=tasks,
                                     smiles_field=self.smiles_field,
                                     featurizer=featurizer,
                                     verbosity="low")
    dataset = data_featurizer.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)

    transformers = []
    metric = Metric(metrics.matthews_corrcoef, np.mean, mode="classification")
    params_dict= {"nb_hidden": [5, 10],
                  "activation": ["relu"],
                  "dropout": [.5],
                  "learning_rate": [.01],
                  "momentum": [.9],
                  "nesterov": [False],
                  "decay": [1e-4],
                  "batch_size": [5],
                  "nb_epoch": [2],
                  "init": ["glorot_uniform"],
                  "nb_layers": [1],
                  "batchnorm": [False],
                  "data_shape": [train_dataset.get_data_shape()]}
    
    optimizer = HyperparamOpt(MultiTaskDNN, tasks, task_types,
                              verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, transformers,
      metric, logdir=None)

  def test_multitask_tf_mlp_ECFP_classification_hyperparam_opt(self):
    """Straightforward test of Tensorflow multitask deepchem classification API."""
    splittype = "scaffold"
    task_type = "classification"

    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: task_type for task in tasks}

    featurizer = CircularFingerprint(size=1024)

    data_featurizer = DataFeaturizer(tasks=tasks,
                                     smiles_field=self.smiles_field,
                                     featurizer=featurizer,
                                     verbosity="low")
    dataset = data_featurizer.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)

    transformers = []

    metric = Metric(metrics.matthews_corrcoef, np.mean, mode="classification")
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
    optimizer = HyperparamOpt(model_builder, tasks, task_types,
                              verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, transformers,
      metric, logdir=None)
