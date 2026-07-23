"""
Tests for hyperopt based hyperparameter optimization.
"""
import unittest
import tempfile
import os
import numpy as np
import pytest
import sklearn
import sklearn.ensemble
import deepchem as dc

try:
    from hyperopt import hp
    has_hyperopt = True
except ModuleNotFoundError:
    has_hyperopt = False


@unittest.skipIf(not has_hyperopt, "hyperopt is not installed")
class TestHyperoptHyperparamOpt(unittest.TestCase):
    """
    Test hyperopt based hyperparameter optimization API.
    """

    def setUp(self):
        """Set up common resources."""

        def rf_model_builder(**model_params):
            rf_params = {
                k: v for (k, v) in model_params.items() if k != 'model_dir'
            }
            model_dir = model_params['model_dir']
            # Fix random_state so that a given set of hyperparameters always
            # yields the same score. This makes the "best model" selection
            # deterministic and the assertions below well defined.
            sklearn_model = sklearn.ensemble.RandomForestRegressor(
                random_state=0, **rf_params)
            return dc.models.SklearnModel(sklearn_model, model_dir)

        self.rf_model_builder = rf_model_builder
        self.max_evals = 5
        self.train_dataset = dc.data.NumpyDataset(X=np.random.rand(50, 5),
                                                  y=np.random.rand(50, 1))
        self.valid_dataset = dc.data.NumpyDataset(X=np.random.rand(20, 5),
                                                  y=np.random.rand(20, 1))

    def test_rf_hyperparam(self):
        """Test of hyperparam_opt with singletask RF regression API (maximize)."""
        optimizer = dc.hyper.HyperoptHyperparamOpt(self.rf_model_builder,
                                                   max_evals=self.max_evals)
        params_dict = {"n_estimators": hp.choice("n_estimators", [10, 100])}
        transformers = []
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict, self.train_dataset, self.valid_dataset, metric,
            transformers)
        valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                          transformers)

        # The returned best model must be the one with the highest score.
        assert valid_score["pearson_r2_score"] == max(all_results.values())
        assert isinstance(best_model, dc.models.SklearnModel)
        assert "n_estimators" in best_hyperparams

    def test_rf_hyperparam_min(self):
        """Test of hyperparam_opt with singletask RF regression API (minimize)."""
        optimizer = dc.hyper.HyperoptHyperparamOpt(self.rf_model_builder,
                                                   max_evals=self.max_evals)
        params_dict = {"n_estimators": hp.choice("n_estimators", [10, 100])}
        transformers = []
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict,
            self.train_dataset,
            self.valid_dataset,
            metric,
            transformers,
            use_max=False)
        valid_score = best_model.evaluate(self.valid_dataset, [metric],
                                          transformers)

        # The returned best model must be the one with the lowest score.
        assert valid_score["pearson_r2_score"] == min(all_results.values())

    def test_rf_with_logdir(self):
        """Test that using a logdir can work correctly."""
        optimizer = dc.hyper.HyperoptHyperparamOpt(self.rf_model_builder,
                                                   max_evals=self.max_evals)
        params_dict = {"n_estimators": hp.choice("n_estimators", [10, 100])}
        transformers = []
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

        with tempfile.TemporaryDirectory() as tmpdirname:
            best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
                params_dict,
                self.train_dataset,
                self.valid_dataset,
                metric,
                transformers,
                logdir=tmpdirname)
            # A results.txt log file and at least one model directory should be
            # written. (hyperopt may re-sample the same discrete value, so the
            # number of unique model directories can be fewer than max_evals.)
            contents = os.listdir(tmpdirname)
            assert "results.txt" in contents
            assert len(contents) >= 2

    @pytest.mark.torch
    def test_multitask_example(self):
        """Test optimizing a multitask torch model with hyperopt."""
        np.random.seed(123)
        train_dataset = dc.data.NumpyDataset(np.random.rand(10, 3),
                                             np.zeros((10, 2)), np.ones(
                                                 (10, 2)), np.arange(10))
        valid_dataset = dc.data.NumpyDataset(np.random.rand(5, 3),
                                             np.zeros((5, 2)), np.ones((5, 2)),
                                             np.arange(5))

        optimizer = dc.hyper.HyperoptHyperparamOpt(
            lambda **params: dc.models.MultitaskRegressor(
                n_tasks=2,
                n_features=3,
                dropouts=[0.],
                weight_init_stddevs=[np.sqrt(6) / np.sqrt(1000)],
                learning_rate=0.003,
                **params),
            max_evals=self.max_evals)

        params_dict = {"batch_size": hp.choice("batch_size", [10, 20])}
        transformers = []
        metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                   task_averager=np.mean)
        best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
            params_dict,
            train_dataset,
            valid_dataset,
            metric,
            transformers,
            use_max=False)

        assert best_model is not None
        assert len(all_results) > 0
        assert "batch_size" in best_hyperparams
