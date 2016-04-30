"""
Integration tests for singletask vector feature models.
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
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.featurize import FeaturizedSamples
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.basic import RDKitDescriptors
from deepchem.featurizers.nnscore import NNScoreComplexFeaturizer
from deepchem.featurizers.grid_featurizer import GridFeaturizer
from deepchem.datasets import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import LogTransformer
from deepchem.transformers import ClippingTransformer
from deepchem.models.test import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
from sklearn.ensemble import RandomForestRegressor

class TestTensorflowAPI(TestAPI):
  """
  Test top-level API for ML models."
  """

  def test_singletask_tf_mlp_ECFP_classification_API(self):
    """Straightforward test of Tensorflow singletask deepchem classification API."""
    splittype = "scaffold"
    output_transformers = []
    input_transformers = []
    task_type = "classification"

    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []

    tasks = ["outcome"]
    task_type = "classification"
    task_types = {task: task_type for task in tasks}
    input_file = "example_classification.csv"
    input_transformers = []
    output_transformers = [NormalizationTransformer]

    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks)

    model_params = {
      "batch_size": 2,
      "num_classification_tasks": 1,
      "num_features": 1024,
      "layer_sizes": [1024],
      "weight_init_stddevs": [1.],
      "bias_init_consts": [0.],
      "dropouts": [.5],
      "num_classes": 2,
      "nb_epoch": 1,
      "penalty": 0.0,
      "optimizer": "adam",
      "learning_rate": .001,
      "data_shape": train_dataset.get_data_shape()
    }
    classification_metrics = [Metric(metrics.roc_auc_score),
                              Metric(metrics.matthews_corrcoef),
                              Metric(metrics.recall_score),
                              Metric(metrics.accuracy_score)]

    model = TensorflowModel(
        tasks, task_types, model_params, self.model_dir,
        tf_class=TensorflowMultiTaskClassifier)
    self._create_model(train_dataset, test_dataset, model, transformers,
                       classification_metrics)
