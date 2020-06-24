"""
Tests for Gaussian Process Hyperparameter Optimization.
"""
import numpy as np
import sklearn
import deepchem as dc
import unittest
import tempfile


class TestGaussianHyperparamOpt(unittest.TestCase):
  """
  Test Gaussian Hyperparameter Optimization.
  """

  def setUp(self):
    """Set up common resources."""

    def rf_model_builder(**model_params):
      rf_params = {k: v for (k, v) in model_params.items() if k != 'model_dir'}
      model_dir = model_params['model_dir']
      sklearn_model = sklearn.ensemble.RandomForestRegressor(**rf_params)
      return dc.models.SklearnModel(sklearn_model, model_dir)

    self.rf_model_builder = rf_model_builder
    self.train_dataset = dc.data.NumpyDataset(
        X=np.random.rand(50, 5), y=np.random.rand(50, 1))
    self.valid_dataset = dc.data.NumpyDataset(
        X=np.random.rand(20, 5), y=np.random.rand(20, 1))

  def test_rf_example(self):
    """Test a simple example of optimizing a RF model with a gaussian process."""

    optimizer = dc.hyper.GaussianProcessHyperparamOpt(self.rf_model_builder)
    params_dict = {"n_estimators": 10}
    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=self.train_dataset)
    ]
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict,
        self.train_dataset,
        self.valid_dataset,
        transformers,
        metric,
        max_iter=2)

    valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                      transformers)
    assert valid_score["pearson_r2_score"] > 0

  def test_rf_with_logdir(self):
    """Test that using a logdir can work correctly."""
    optimizer = dc.hyper.GaussianProcessHyperparamOpt(self.rf_model_builder)
    params_dict = {"n_estimators": 10}
    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=self.train_dataset)
    ]
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    with tempfile.TemporaryDirectory() as tmpdirname:
      best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
          params_dict,
          self.train_dataset,
          self.valid_dataset,
          transformers,
          metric,
          logdir=tmpdirname,
          max_iter=2)
    valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                      transformers)
    assert valid_score["pearson_r2_score"] > 0
