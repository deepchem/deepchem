"""
Tests for hyperparam optimization.
"""

import unittest
import sklearn
import deepchem as dc


class TestHyperparamOpt(unittest.TestCase):
    """
    Test abstract superclass behavior.
    """

    def test_cant_be_initialized(self):
        """Test HyperparamOpt can't be initialized."""
        initialized = True

        def rf_model_builder(model_params, model_dir):
            sklearn_model = sklearn.ensemble.RandomForestRegressor(
                **model_params)
            return dc.model.SklearnModel(sklearn_model, model_dir)

        try:
            _ = dc.hyper.HyperparamOpt(rf_model_builder)
        except ValueError:
            initialized = False
        assert not initialized
