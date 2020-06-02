"""
Tests for hyperparam optimization.
"""
import os
import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


class TestHyperparamOpt(unittest.TestCase):
  """
  Test abstract superclass behavior.
  """

  def test_cant_be_initialized(self):
    """Test HyperparamOpt can't be initialized."""
    initialized = True

    def rf_model_builder(model_params, model_dir):
      sklearn_model = sklearn.ensemble.RandomForestRegressor(**model_params)
      return dc.model.SklearnModel(sklearn_model, model_dir)

    try:
      opt = dc.hyper.HyperparamOpt(rf_model_builder)
    except:
      initialized = False
    assert not initialized


class TestGridHyperparamOpt(unittest.TestCase):
  """
  Test grid hyperparameter optimization API.
  """

  def test_singletask_sklearn_rf_ECFP_regression_hyperparam_opt(self):
    """Test of hyperparam_opt with singletask RF ECFP regression API."""
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "../../models/tests/example.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)

    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    params_dict = {"n_estimators": [10, 100]}
    metric = dc.metrics.Metric(dc.metrics.r2_score)

    def rf_model_builder(model_params, model_dir):
      sklearn_model = RandomForestRegressor(**model_params)
      return dc.models.SklearnModel(sklearn_model, model_dir)

    optimizer = dc.hyper.GridHyperparamOpt(rf_model_builder)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict,
        train_dataset,
        valid_dataset,
        transformers,
        metric,
        logdir=None)

  def test_singletask_to_multitask_sklearn_hyperparam_opt(self):
    """Test of hyperparam_opt with singletask_to_multitask."""
    tasks = [
        "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
        "task8", "task9", "task10", "task11", "task12", "task13", "task14",
        "task15", "task16"
    ]
    input_file = "multitask_example.csv"

    n_features = 10
    n_tasks = len(tasks)
    # Define train dataset
    n_train = 100
    X_train = np.random.rand(n_train, n_features)
    y_train = np.random.randint(2, size=(n_train, n_tasks))
    w_train = np.ones_like(y_train)
    ids_train = ["C"] * n_train

    train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train, w_train,
                                                   ids_train, tasks)

    # Define validation dataset
    n_valid = 10
    X_valid = np.random.rand(n_valid, n_features)
    y_valid = np.random.randint(2, size=(n_valid, n_tasks))
    w_valid = np.ones_like(y_valid)
    ids_valid = ["C"] * n_valid
    valid_dataset = dc.data.DiskDataset.from_numpy(X_valid, y_valid, w_valid,
                                                   ids_valid, tasks)

    transformers = []
    classification_metric = dc.metrics.Metric(
        dc.metrics.matthews_corrcoef, np.mean, mode="classification")
    params_dict = {"n_estimators": [1, 10]}

    def multitask_model_builder(model_params, model_dir):

      def model_builder(model_dir):
        sklearn_model = RandomForestClassifier(**model_params)
        return dc.models.SklearnModel(sklearn_model, model_dir)

      return dc.models.SingletaskToMultitask(tasks, model_builder, model_dir)

    optimizer = dc.hyper.GridHyperparamOpt(multitask_model_builder)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict,
        train_dataset,
        valid_dataset,
        transformers,
        classification_metric,
        logdir=None)

  def test_multitask_tf_mlp_ECFP_classification_hyperparam_opt(self):
    """Straightforward test of Tensorflow multitask deepchem classification API."""
    task_type = "classification"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir,
                              "../../models/tests/multitask_example.csv")
    tasks = [
        "task0", "task1", "task2", "task3", "task4", "task5", "task6", "task7",
        "task8", "task9", "task10", "task11", "task12", "task13", "task14",
        "task15", "task16"
    ]

    n_features = 1024
    featurizer = dc.feat.CircularFingerprint(size=n_features)

    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)

    transformers = []
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
    params_dict = {"layer_sizes": [(10,), (100,)]}

    def model_builder(model_params, model_dir):
      return dc.models.MultitaskClassifier(
          len(tasks), n_features, model_dir=model_dir, **model_params)

    optimizer = dc.hyper.GridHyperparamOpt(model_builder)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict,
        train_dataset,
        valid_dataset,
        transformers,
        metric,
        logdir=None)
