"""
Tests to make sure deepchem models can fit models on easy datasets.
"""

import tempfile
import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_digits
from sklearn.model_selection import train_test_split
try:
    import xgboost
    import lightgbm
    has_xgboost_and_lightgbm = True
except:
    has_xgboost_and_lightgbm = False

import deepchem as dc


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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
    xgb_model = xgboost.XGBRegressor(n_estimators=50,
                                     random_state=123,
                                     verbose=False)
    model = dc.models.GBDTModel(xgb_model, **params)
    # fit trained model
    model.fit(train_dataset)
    model.save()
    # eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    assert scores[regression_metric.name] < 55


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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
    lgbm_model = lightgbm.LGBMRegressor(n_estimators=50,
                                        random_state=123,
                                        silent=True)
    model = dc.models.GBDTModel(lgbm_model, **params)
    # fit trained model
    model.fit(train_dataset)
    model.save()
    # eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    assert scores[regression_metric.name] < 55


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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
        xgb_model = xgboost.XGBRegressor(n_estimators=50,
                                         seed=123,
                                         verbose=False)
        return dc.models.GBDTModel(xgb_model, model_dir, **params)

    model = dc.models.SingletaskToMultitask(tasks, xgboost_builder)
    # fit trained model
    model.fit(train_dataset)
    model.save()
    # eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    score = scores[regression_metric.name]
    assert score < 55


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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
        lgbm_model = lightgbm.LGBMRegressor(n_estimators=50,
                                            seed=123,
                                            silent=False)
        return dc.models.GBDTModel(lgbm_model, model_dir, **params)

    model = dc.models.SingletaskToMultitask(tasks, lightgbm_builder)
    # fit trained model
    model.fit(train_dataset)
    model.save()
    # eval model on test
    scores = model.evaluate(test_dataset, [regression_metric])
    score = scores[regression_metric.name]
    assert score < 55


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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
    xgb_model = xgboost.XGBRegressor(n_estimators=50,
                                     random_state=123,
                                     verbose=False)
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


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
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
    lgbm_model = lightgbm.LGBMRegressor(n_estimators=50,
                                        random_state=123,
                                        silent=True)
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


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
def test_earlystopping_with_xgboost():
    np.random.seed(123)

    # prepare dataset
    N_samples = 50000
    n_features = 1000
    X = np.random.rand(N_samples, n_features)
    y = np.random.rand(N_samples)
    dataset = dc.data.NumpyDataset(X, y)

    # xgboost test
    xgb_model = xgboost.XGBRegressor(n_estimators=20, random_state=123)
    model = dc.models.GBDTModel(xgb_model, early_stopping_rounds=3)
    # fit trained model
    model.fit(dataset)

    # If ES rounds are more than total epochs, it will never trigger
    if model.early_stopping_rounds < model.model.n_estimators:
        # Find the number of boosting rounds in the model
        res = list(model.model.evals_result_['validation_0'].values())
        rounds_boosted = len(res[0])
        # If rounds boosted are less than total estimators, it means ES was triggered
        if rounds_boosted < model.model.n_estimators:
            assert model.model.best_iteration < model.model.n_estimators - 1


@unittest.skipIf(not has_xgboost_and_lightgbm,
                 'xgboost or lightgbm are not installed')
def test_earlystopping_with_lightgbm():
    np.random.seed(123)

    # prepare dataset
    N_samples = 50000
    n_features = 1000
    X = np.random.rand(N_samples, n_features)
    y = np.random.rand(N_samples)
    dataset = dc.data.NumpyDataset(X, y)

    # lightgbm test
    lgbm_model = lightgbm.LGBMRegressor(n_estimators=20,
                                        random_state=123,
                                        silent=True)
    model = dc.models.GBDTModel(lgbm_model, early_stopping_rounds=3)
    # fit trained model
    model.fit(dataset)

    # If ES rounds are more than total epochs, it will never trigger
    if model.early_stopping_rounds < model.model.n_estimators:
        # Find the number of boosting rounds in the model
        res = list(model.model.evals_result_['valid_0'].values())
        rounds_ran = len(res[0])
        # If rounds ran are less than estimators, it means ES was triggered
        if rounds_ran < model.model.n_estimators:
            assert model.model.best_iteration_ < model.model.n_estimators
