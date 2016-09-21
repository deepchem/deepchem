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
from deepchem.featurizers.featurize import DataLoader
from deepchem.transformers import NormalizationTransformer
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.multitask import SingletaskToMultitask 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from deepchem.datasets import Dataset
from deepchem.hyperparameters import HyperparamOpt
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.keras_models import KerasModel
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.splits import ScaffoldSplitter
import tensorflow as tf
from keras import backend as K

class TestHyperparamOptAPI(TestAPI):
  """
  Test hyperparameter optimization API.
  """
  def test_singletask_sklearn_rf_ECFP_regression_hyperparam_opt(self):
    """Test of hyperparam_opt with singletask RF ECFP regression API."""
    featurizer = CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    input_file = os.path.join(self.current_dir, "example.csv")
    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)

    transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        transformer.transform(dataset)

    params_dict = {"n_estimators": [10, 100]}
    metric = Metric(metrics.r2_score)
    def rf_model_builder(model_params, model_dir):
      sklearn_model = RandomForestRegressor(**model_params)
      return SklearnModel(sklearn_model, model_dir)

    optimizer = HyperparamOpt(rf_model_builder, verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, transformers,
      metric, logdir=None)

  def test_singletask_to_multitask_sklearn_hyperparam_opt(self):
    """Test of hyperparam_opt with singletask_to_multitask."""
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
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

    transformers = []
    classification_metric = Metric(metrics.matthews_corrcoef, np.mean,
                                   mode="classification")
    params_dict = {"n_estimators": [1, 10]}
    def multitask_model_builder(model_params, model_dir):
      def model_builder(model_dir):
        sklearn_model = RandomForestClassifier(**model_params)
        return SklearnModel(sklearn_model, model_dir)
      return SingletaskToMultitask(tasks, model_builder, model_dir)

    optimizer = HyperparamOpt(multitask_model_builder, verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, transformers,
      classification_metric, logdir=None)

  def test_multitask_keras_mlp_ECFP_classification_hyperparam_opt(self):
    """Straightforward test of Keras multitask deepchem classification API."""
    task_type = "classification"
    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]

    n_features = 1024
    featurizer = CircularFingerprint(size=n_features)
    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)

    transformers = []
    metric = Metric(metrics.matthews_corrcoef, np.mean, mode="classification")
    params_dict= {"n_hidden": [5, 10]}
      
    def model_builder(model_params, model_dir):
      keras_model = MultiTaskDNN(
          len(tasks), n_features, task_type, dropout=0., **model_params)
      return KerasModel(keras_model, model_dir)
    optimizer = HyperparamOpt(model_builder, verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, transformers,
      metric, logdir=None)

  def test_multitask_tf_mlp_ECFP_classification_hyperparam_opt(self):
    """Straightforward test of Tensorflow multitask deepchem classification API."""
    task_type = "classification"

    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]

    n_features = 1024
    featurizer = CircularFingerprint(size=n_features)

    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, self.train_dir, self.valid_dir, self.test_dir)


    transformers = []
    metric = Metric(metrics.matthews_corrcoef, np.mean, mode="classification")
    params_dict = {"layer_sizes": [(10,), (100,)]}

    def model_builder(model_params, model_dir):
        tensorflow_model = TensorflowMultiTaskClassifier(
            len(tasks), n_features, model_dir, **model_params)
        return TensorflowModel(tensorflow_model, model_dir)
    optimizer = HyperparamOpt(model_builder, verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, transformers, metric,
      logdir=None)
