"""
Integration tests for hyperparam optimization.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
import tempfile
import shutil
from deepchem.models.test import TestAPI
from deepchem.models.sklearn_models import SklearnModel
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import NormalizationTransformer
from deepchem import metrics
from deepchem.metrics import Metric
from sklearn.ensemble import RandomForestRegressor

def rf_model_builder(task_types, params_dict, logdir=None,
                     train=True):
    """Builds random forests given hyperparameters.

    Last two arguments only for tensorflow models and ignored.
    """
    n_estimators = params_dict["n_estimators"]
    max_features = params_dict["max_features"]
    return SklearnModel(
        task_types, params_dict,
        model_instance=RandomForestRegressor(n_estimators=n_estimators,
                                             max_features=max_features))

class TestHyperparamOptAPI(TestAPI):
  """
  Test hyperparameter optimization API.
  """
  def test_singletask_sklearn_rf_ECFP_regression_hyperparam_opt(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformer_classes = []
    output_transformer_classes = [NormalizationTransformer]
    task_types = {"log-solubility": "regression"}
    input_file = "example.csv"
    train_dataset, valid_dataset, _, output_transformers, = \
        self._featurize_train_test_split(
            splittype, compound_featurizers, 
            complex_featurizers, input_transformer_classes,
            output_transformer_classes, input_file, task_types.keys())
    params_dict = {
      "n_estimators": [10, 100],
      "max_features": ["auto"],
      "data_shape": train_dataset.get_data_shape()
    }
    metric = Metric(metrics.r2_score)

    self._hyperparam_opt(rf_model_builder, params_dict, train_dataset,
                         valid_dataset, output_transformers, task_types,
                         metric)
