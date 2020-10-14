"""
Tests to make sure deepchem models can fit models on easy datasets.
"""

import sklearn
import sklearn.datasets
import numpy as np
import deepchem as dc
import xgboost
import lightgbm


def test_xgboost_regression():
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

  xgb_model = xgboost.XGBRegressor(
      n_estimators=50, random_state=123, verbose=False)
  model = dc.models.GBDTModel(xgb_model, **esr)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  assert scores[regression_metric.name] < 55


def test_xgboost_multitask_regression():
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
    xgb_model = xgboost.XGBRegressor(n_estimators=50, seed=123, verbose=False)
    return dc.models.GBDTModel(xgb_model, model_dir, **esr)

  model = dc.models.SingletaskToMultitask(tasks, model_builder)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  score = scores[regression_metric.name]
  assert score < 55


def test_xgboost_classification():
  """Test that sklearn models can learn on simple classification datasets."""
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
  xgb_model = xgboost.XGBClassifier(n_estimators=50, seed=123, verbose=False)
  model = dc.models.GBDTModel(xgb_model, **esr)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on test
  scores = model.evaluate(test_dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_lightgbm_regression():
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

  lgbm_model = lightgbm.LGBMRegressor(
      n_estimators=50, random_state=123, silent=True)
  model = dc.models.GBDTModel(lgbm_model, **esr)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  assert scores[regression_metric.name] < 55


def test_lightgbm_multitask_regression():
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
    lgbm_model = lightgbm.LGBMRegressor(n_estimators=50, seed=123, silent=True)
    return dc.models.GBDTModel(lgbm_model, model_dir, **esr)

  model = dc.models.SingletaskToMultitask(tasks, model_builder)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  score = scores[regression_metric.name]
  assert score < 55


def test_lightgbm_classification():
  """Test that sklearn models can learn on simple classification datasets."""
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
  lgbm_model = lightgbm.LGBMClassifier(n_estimators=50, seed=123, silent=True)
  model = dc.models.GBDTModel(lgbm_model, **esr)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  # Eval model on test
  scores = model.evaluate(test_dataset, [classification_metric])
  assert scores[classification_metric.name] > .9
