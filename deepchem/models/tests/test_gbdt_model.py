"""
Tests to make sure deepchem models can fit models on easy datasets.
"""

import tempfile

import numpy as np
import xgboost
import lightgbm
from sklearn.datasets import load_diabetes, load_digits
from sklearn.model_selection import train_test_split

import deepchem as dc


def test_singletask_regression_with_xgboost():
  np.random.seed(123)

  # prepare dataset
  dataset = load_diabetes()
  X, y = dataset.data, dataset.target
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.NumpyDataset(X_train, y_train)
  test_dataset = dc.data.NumpyDataset(X_test, y_test)

  # global setting
  regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
  params = {'early_stopping_rounds': 25}

  # xgboost test
  xgb_model = xgboost.XGBRegressor(
      n_estimators=50, random_state=123, verbose=False)
  model = dc.models.GBDTModel(xgb_model, **params)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  assert scores[regression_metric.name] < 55


def test_singletask_regression_with_lightgbm():
  np.random.seed(123)

  # prepare dataset
  dataset = load_diabetes()
  X, y = dataset.data, dataset.target
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.NumpyDataset(X_train, y_train)
  test_dataset = dc.data.NumpyDataset(X_test, y_test)

  # global setting
  regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
  params = {'early_stopping_rounds': 25}

  # lightgbm test
  lgbm_model = lightgbm.LGBMRegressor(
      n_estimators=50, random_state=123, silent=True)
  model = dc.models.GBDTModel(lgbm_model, **params)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  assert scores[regression_metric.name] < 55


def test_multitask_regression_with_xgboost():
  np.random.seed(123)

  # prepare dataset
  n_tasks = 4
  tasks = range(n_tasks)
  dataset = load_diabetes()
  X, y = dataset.data, dataset.target
  y = np.reshape(y, (len(y), 1))
  y = np.hstack([y] * n_tasks)
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
  test_dataset = dc.data.DiskDataset.from_numpy(X_test, y_test)

  # global setting
  regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
  params = {'early_stopping_rounds': 25}

  # xgboost test
  def xgboost_builder(model_dir):
    xgb_model = xgboost.XGBRegressor(n_estimators=50, seed=123, verbose=False)
    return dc.models.GBDTModel(xgb_model, model_dir, **params)

  model = dc.models.SingletaskToMultitask(tasks, xgboost_builder)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  score = scores[regression_metric.name]
  assert score < 55


def test_multitask_regression_with_lightgbm():
  np.random.seed(123)

  # prepare dataset
  n_tasks = 4
  tasks = range(n_tasks)
  dataset = load_diabetes()
  X, y = dataset.data, dataset.target
  y = np.reshape(y, (len(y), 1))
  y = np.hstack([y] * n_tasks)
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
  test_dataset = dc.data.DiskDataset.from_numpy(X_test, y_test)

  # global setting
  regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
  params = {'early_stopping_rounds': 25}

  # lightgbm test
  def lightgbm_builder(model_dir):
    lgbm_model = lightgbm.LGBMRegressor(n_estimators=50, seed=123, silent=False)
    return dc.models.GBDTModel(lgbm_model, model_dir, **params)

  model = dc.models.SingletaskToMultitask(tasks, lightgbm_builder)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # eval model on test
  scores = model.evaluate(test_dataset, [regression_metric])
  score = scores[regression_metric.name]
  assert score < 55


def test_classification_with_xgboost():
  """Test that sklearn models can learn on simple classification datasets."""
  np.random.seed(123)

  # prepare dataset
  dataset = load_digits(n_class=2)
  X, y = dataset.data, dataset.target
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.NumpyDataset(X_train, y_train)
  test_dataset = dc.data.NumpyDataset(X_test, y_test)

  # global setting
  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  params = {'early_stopping_rounds': 25}

  # xgboost test
  xgb_model = xgboost.XGBClassifier(n_estimators=50, seed=123, verbose=False)
  model = dc.models.GBDTModel(xgb_model, **params)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # eval model on test
  scores = model.evaluate(test_dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_classification_with_lightgbm():
  """Test that sklearn models can learn on simple classification datasets."""
  np.random.seed(123)

  # prepare dataset
  dataset = load_digits(n_class=2)
  X, y = dataset.data, dataset.target
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.NumpyDataset(X_train, y_train)
  test_dataset = dc.data.NumpyDataset(X_test, y_test)

  # global setting
  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
  params = {'early_stopping_rounds': 25}

  # lightgbm test
  lgbm_model = lightgbm.LGBMClassifier(n_estimators=50, seed=123, silent=True)
  model = dc.models.GBDTModel(lgbm_model, **params)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # eval model on test
  scores = model.evaluate(test_dataset, [classification_metric])
  assert scores[classification_metric.name] > .9


def test_reload_with_xgboost():
  np.random.seed(123)

  # prepare dataset
  dataset = load_diabetes()
  X, y = dataset.data, dataset.target
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.NumpyDataset(X_train, y_train)
  test_dataset = dc.data.NumpyDataset(X_test, y_test)

  # global setting
  regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
  model_dir = tempfile.mkdtemp()
  params = {'early_stopping_rounds': 25, 'model_dir': model_dir}

  # xgboost test
  xgb_model = xgboost.XGBRegressor(
      n_estimators=50, random_state=123, verbose=False)
  model = dc.models.GBDTModel(xgb_model, **params)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # reload
  reloaded_model = dc.models.GBDTModel(None, model_dir)
  reloaded_model.reload()
  # check predictions match on test dataset
  original_pred = model.predict(test_dataset)
  reload_pred = reloaded_model.predict(test_dataset)
  assert np.all(original_pred == reload_pred)
  # eval model on test
  scores = reloaded_model.evaluate(test_dataset, [regression_metric])
  assert scores[regression_metric.name] < 55


def test_reload_with_lightgbm():
  np.random.seed(123)

  # prepare dataset
  dataset = load_diabetes()
  X, y = dataset.data, dataset.target
  frac_train = .7
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=frac_train)
  train_dataset = dc.data.NumpyDataset(X_train, y_train)
  test_dataset = dc.data.NumpyDataset(X_test, y_test)

  # global setting
  regression_metric = dc.metrics.Metric(dc.metrics.mae_score)
  model_dir = tempfile.mkdtemp()
  params = {'early_stopping_rounds': 25, 'model_dir': model_dir}

  # lightgbm test
  lgbm_model = lightgbm.LGBMRegressor(
      n_estimators=50, random_state=123, silent=True)
  model = dc.models.GBDTModel(lgbm_model, **params)
  # fit trained model
  model.fit(train_dataset)
  model.save()
  # reload
  reloaded_model = dc.models.GBDTModel(None, model_dir)
  reloaded_model.reload()
  # check predictions match on test dataset
  original_pred = model.predict(test_dataset)
  reload_pred = reloaded_model.predict(test_dataset)
  assert np.all(original_pred == reload_pred)
  # eval model on test
  scores = reloaded_model.evaluate(test_dataset, [regression_metric])
  assert scores[regression_metric.name] < 55
