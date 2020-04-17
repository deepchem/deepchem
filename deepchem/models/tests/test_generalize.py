"""
Tests to make sure deepchem models can fit models on easy datasets.
"""

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import sklearn
import sklearn.datasets
import numpy as np
import unittest
import tempfile
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


class TestGeneralize(unittest.TestCase):
  """
  Test that models can learn generalizable models on simple datasets.
  """

  def test_sklearn_regression(self):
    """Test that sklearn models can learn on simple regression datasets."""
    np.random.seed(123)

    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.expand_dims(y, 1)
    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)

    sklearn_model = LinearRegression()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    assert scores[regression_metric.name] > .5

  def test_sklearn_transformed_regression(self):
    """Test that sklearn models can learn on simple transformed regression datasets."""
    np.random.seed(123)
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.expand_dims(y, 1)

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    # Eval model on train
    transformers = [
        dc.trans.NormalizationTransformer(
            transform_X=True, dataset=train_dataset),
        dc.trans.ClippingTransformer(transform_X=True, dataset=train_dataset),
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    for data in [train_dataset, test_dataset]:
      for transformer in transformers:
        data = transformer.transform(data)

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)
    sklearn_model = LinearRegression()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    train_scores = model.evaluate(train_dataset, [regression_metric],
                                  transformers)
    assert train_scores[regression_metric.name] > .5

    # Eval model on test
    test_scores = model.evaluate(test_dataset, [regression_metric],
                                 transformers)
    assert test_scores[regression_metric.name] > .5

  def test_sklearn_multitask_regression(self):
    """Test that sklearn models can learn on simple multitask regression."""
    np.random.seed(123)
    n_tasks = 4
    tasks = range(n_tasks)
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.reshape(y, (len(y), 1))
    y = np.hstack([y] * n_tasks)

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
    test_dataset = dc.data.DiskDataset.from_numpy(X_test, y_test)

    regression_metric = dc.metrics.Metric(dc.metrics.r2_score)

    def model_builder(model_dir):
      sklearn_model = LinearRegression()
      return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(tasks, model_builder)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    for score in scores[regression_metric.name]:
      assert score > .5

  #def test_sklearn_classification(self):
  #  """Test that sklearn models can learn on simple classification datasets."""
  #  np.random.seed(123)
  #  dataset = sklearn.datasets.load_digits(n_class=2)
  #  X, y = dataset.data, dataset.target

  #  frac_train = .7
  #  n_samples = len(X)
  #  n_train = int(frac_train*n_samples)
  #  X_train, y_train = X[:n_train], y[:n_train]
  #  X_test, y_test = X[n_train:], y[n_train:]
  #  train_dataset = dc.data.NumpyDataset(X_train, y_train)
  #  test_dataset = dc.data.NumpyDataset(X_test, y_test)

  #  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  #  sklearn_model = LogisticRegression()
  #  model = dc.models.SklearnModel(sklearn_model)

  #  # Fit trained model
  #  model.fit(train_dataset)
  #  model.save()

  #  # Eval model on test
  #  scores = model.evaluate(test_dataset, [classification_metric])
  #  assert scores[classification_metric.name] > .5

  #def test_sklearn_multitask_classification(self):
  #  """Test that sklearn models can learn on simple multitask classification."""
  #  np.random.seed(123)
  #  n_tasks = 4
  #  tasks = range(n_tasks)
  #  dataset = sklearn.datasets.load_digits(n_class=2)
  #  X, y = dataset.data, dataset.target
  #  y = np.reshape(y, (len(y), 1))
  #  y = np.hstack([y] * n_tasks)
  #
  #  frac_train = .7
  #  n_samples = len(X)
  #  n_train = int(frac_train*n_samples)
  #  X_train, y_train = X[:n_train], y[:n_train]
  #  X_test, y_test = X[n_train:], y[n_train:]
  #  train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
  #  test_dataset = dc.data.DiskDataset.from_numpy(X_test, y_test)

  #  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  #  def model_builder(model_dir):
  #    sklearn_model = LogisticRegression()
  #    return dc.models.SklearnModel(sklearn_model, model_dir)
  #  model = dc.models.SingletaskToMultitask(tasks, model_builder)

  #  # Fit trained model
  #  model.fit(train_dataset)
  #  model.save()
  #  # Eval model on test
  #  scores = model.evaluate(test_dataset, [classification_metric])
  #  for score in scores[classification_metric.name]:
  #    assert score > .5

  def test_xgboost_regression(self):
    import xgboost
    np.random.seed(123)

    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
    # Set early stopping round = n_estimators so that esr won't work
    esr = {'early_stopping_rounds': 50}

    xgb_model = xgboost.XGBRegressor(n_estimators=50, random_state=123)
    model = dc.models.XGBoostModel(xgb_model, **esr)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    assert scores[regression_metric.name] < 55

  def test_xgboost_multitask_regression(self):
    import xgboost
    np.random.seed(123)
    n_tasks = 4
    tasks = range(n_tasks)
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target
    y = np.reshape(y, (len(y), 1))
    y = np.hstack([y] * n_tasks)

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
    test_dataset = dc.data.DiskDataset.from_numpy(X_test, y_test)

    regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
    esr = {'early_stopping_rounds': 50}

    def model_builder(model_dir):
      xgb_model = xgboost.XGBRegressor(n_estimators=50, seed=123)
      return dc.models.XGBoostModel(xgb_model, model_dir, **esr)

    model = dc.models.SingletaskToMultitask(tasks, model_builder)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    for score in scores[regression_metric.name]:
      assert score < 50

  def test_xgboost_classification(self):
    """Test that sklearn models can learn on simple classification datasets."""
    import xgboost
    np.random.seed(123)
    dataset = sklearn.datasets.load_digits(n_class=2)
    X, y = dataset.data, dataset.target

    frac_train = .7
    n_samples = len(X)
    n_train = int(frac_train * n_samples)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    esr = {'early_stopping_rounds': 50}
    xgb_model = xgboost.XGBClassifier(n_estimators=50, seed=123)
    model = dc.models.XGBoostModel(xgb_model, **esr)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on test
    scores = model.evaluate(test_dataset, [classification_metric])
    assert scores[classification_metric.name] > .9
