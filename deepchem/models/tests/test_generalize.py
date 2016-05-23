"""
Tests to make sure deepchem models can fit models on easy datasets.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import sklearn
import sklearn.datasets
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from deepchem.datasets import Dataset
from deepchem.metrics import Metric
from deepchem.models.tests import TestAPI
from deepchem.models.sklearn_models import SklearnModel
from deepchem import metrics
from deepchem.utils.evaluate import Evaluator
from deepchem.models.multitask import SingletaskToMultitask
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import ClippingTransformer

class TestGeneralization(TestAPI):
  """
  Test that models can learn generalizable models on simple datasets.
  """

  def test_sklearn_regression(self):
    """Test that sklearn models can learn on simple regression datasets."""
    np.random.seed(123)
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target

    frac_train = .7
    n_samples = len(X)
    
    X_train, y_train = X[:frac_train*n_samples], y[:frac_train*n_samples]
    X_test, y_test = X[frac_train*n_samples:], y[frac_train*n_samples:]

    train_dataset = Dataset.from_numpy(self.train_dir, X_train, y_train)
    test_dataset = Dataset.from_numpy(self.test_dir, X_test, y_test)

    tasks = train_dataset.get_task_names()
    task_types = {task: "regression" for task in tasks}

    model_params = {
      "batch_size": None,
      "data_shape": train_dataset.get_data_shape()
    }

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="regression",
                         model_instance=LinearRegression())

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    transformers = []
    train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
    train_scores = train_evaluator.compute_model_performance([regression_metric])
    print("train_scores")
    print(train_scores)

    # Eval model on test
    transformers = []
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])
    print("scores")
    print(scores)

    assert scores[regression_metric.name] > .5

  def test_sklearn_transformed_regression(self):
    """Test that sklearn models can learn on simple transformed regression datasets."""
    np.random.seed(123)
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target

    frac_train = .7
    n_samples = len(X)
    
    X_train, y_train = X[:frac_train*n_samples], y[:frac_train*n_samples]
    X_test, y_test = X[frac_train*n_samples:], y[frac_train*n_samples:]

    train_dataset = Dataset.from_numpy(self.train_dir, X_train, y_train)
    test_dataset = Dataset.from_numpy(self.test_dir, X_test, y_test)


    # Eval model on train
    input_transformers = [
        NormalizationTransformer(transform_X=True, dataset=train_dataset),
        ClippingTransformer(transform_X=True, dataset=train_dataset)]
    output_transformers = [
      NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    transformers = input_transformers + output_transformers
    for transformer in transformers:
        transformer.transform(train_dataset)
    for transformer in transformers:
        transformer.transform(test_dataset)

    tasks = train_dataset.get_task_names()
    task_types = {task: "regression" for task in tasks}

    model_params = {
      "batch_size": None,
      "data_shape": train_dataset.get_data_shape()
    }

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="regression",
                         model_instance=LinearRegression())

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
    train_scores = train_evaluator.compute_model_performance([regression_metric])
    print("train_scores")
    print(train_scores)

    assert train_scores[regression_metric.name] > .5

    # Eval model on test
    transformers = []
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])
    print("scores")
    print(scores)

    assert scores[regression_metric.name] > .5

  def test_sklearn_multitask_regression(self):
    """Test that sklearn models can learn on simple multitask regression."""
    np.random.seed(123)
    n_tasks = 4
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.reshape(y, (len(y), 1))
    y = np.hstack([y] * n_tasks)
    
    frac_train = .7
    n_samples = len(X)
    
    X_train, y_train = X[:frac_train*n_samples], y[:frac_train*n_samples]
    X_test, y_test = X[frac_train*n_samples:], y[frac_train*n_samples:]

    train_dataset = Dataset.from_numpy(self.train_dir, X_train, y_train)
    test_dataset = Dataset.from_numpy(self.test_dir, X_test, y_test)

    tasks = train_dataset.get_task_names()
    task_types = {task: "regression" for task in tasks}

    model_params = {
      "batch_size": None,
      "data_shape": train_dataset.get_data_shape()
    }

    verbosity = "high"
    regression_metric = Metric(metrics.r2_score, verbosity=verbosity)
    def model_builder(tasks, task_types, model_params, model_dir, verbosity=None):
      return SklearnModel(tasks, task_types, model_params, model_dir,
                          mode="regression",
                          model_instance=LinearRegression(),
                          verbosity=verbosity)
    model = SingletaskToMultitask(tasks, task_types, model_params, self.model_dir,
                                  model_builder, verbosity=verbosity)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    transformers = []
    train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
    train_scores = train_evaluator.compute_model_performance([regression_metric])
    print("train_scores")
    print(train_scores)

    # Eval model on test
    transformers = []
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([regression_metric])
    print("scores")
    print(scores)

    for score in scores[regression_metric.name]:
      assert score > .5

  def test_sklearn_classification(self):
    """Test that sklearn models can learn on simple classification datasets."""
    np.random.seed(123)
    dataset = sklearn.datasets.load_digits(n_class=2)
    X, y = dataset.data, dataset.target

    frac_train = .7
    n_samples = len(X)
    
    X_train, y_train = X[:frac_train*n_samples], y[:frac_train*n_samples]
    X_test, y_test = X[frac_train*n_samples:], y[frac_train*n_samples:]

    print("X_train.shape, y_train.shape, X_test.shape, y_test.shape")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_dataset = Dataset.from_numpy(self.train_dir, X_train, y_train)
    test_dataset = Dataset.from_numpy(self.test_dir, X_test, y_test)

    tasks = train_dataset.get_task_names()
    task_types = {task: "classification" for task in tasks}

    model_params = {
      "batch_size": None,
      "data_shape": train_dataset.get_data_shape()
    }

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="classification",
                         model_instance=LogisticRegression())

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    transformers = []
    train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
    train_scores = train_evaluator.compute_model_performance([classification_metric])
    print("train_scores")
    print(train_scores)

    # Eval model on test
    transformers = []
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])
    print("scores")
    print(scores)

    assert scores[classification_metric.name] > .5

  def test_sklearn_multitask_classification(self):
    """Test that sklearn models can learn on simple multitask classification."""
    np.random.seed(123)
    n_tasks = 4
    dataset = sklearn.datasets.load_digits(n_class=2)
    X, y = dataset.data, dataset.target
    y = np.reshape(y, (len(y), 1))
    y = np.hstack([y] * n_tasks)
    
    frac_train = .7
    n_samples = len(X)
    
    X_train, y_train = X[:frac_train*n_samples], y[:frac_train*n_samples]
    X_test, y_test = X[frac_train*n_samples:], y[frac_train*n_samples:]

    train_dataset = Dataset.from_numpy(self.train_dir, X_train, y_train)
    test_dataset = Dataset.from_numpy(self.test_dir, X_test, y_test)

    tasks = train_dataset.get_task_names()
    task_types = {task: "classification" for task in tasks}

    model_params = {
      "batch_size": None,
      "data_shape": train_dataset.get_data_shape()
    }

    verbosity = "high"
    classification_metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    def model_builder(tasks, task_types, model_params, model_dir, verbosity=None):
      return SklearnModel(tasks, task_types, model_params, model_dir,
                          mode="classification",
                          model_instance=LogisticRegression(),
                          verbosity=verbosity)
    model = SingletaskToMultitask(tasks, task_types, model_params, self.model_dir,
                                  model_builder, verbosity=verbosity)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    transformers = []
    train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
    train_scores = train_evaluator.compute_model_performance([classification_metric])
    print("train_scores")
    print(train_scores)

    # Eval model on test
    transformers = []
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=verbosity)
    scores = evaluator.compute_model_performance([classification_metric])
    print("scores")
    print(scores)

    for score in scores[classification_metric.name]:
      assert score > .5
