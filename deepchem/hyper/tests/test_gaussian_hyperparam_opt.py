"""
Tests for Gaussian Process Hyperparameter Optimization.
"""
import numpy as np
import sklearn
import deepchem as dc
import unittest


class TestGaussianHyperparamOpt(unittest.TestCase):
  """
  Test Gaussian Hyperparameter Optimization.
  """

  def test_rf_example(self):

    def rf_model_builder(model_params, model_dir):
      sklearn_model = sklearn.ensemble.RandomForestRegressor(**model_params)
      return dc.models.SklearnModel(sklearn_model, model_dir)

    train_dataset = dc.data.NumpyDataset(
        X=np.random.rand(50, 5), y=np.random.rand(50, 1))
    valid_dataset = dc.data.NumpyDataset(
        X=np.random.rand(20, 5), y=np.random.rand(20, 1))
    optimizer = dc.hyper.GaussianProcessHyperparamOpt(rf_model_builder)
    params_dict = {"n_estimators": 40}
    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    metric = dc.metrics.Metric(dc.metrics.r2_score)

    best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, transformers, metric)

    ########################################
    print("best_hyperparams")
    print(best_hyperparams)
    print("all_results")
    print(all_results)
    assert 0 == 1
    ########################################
