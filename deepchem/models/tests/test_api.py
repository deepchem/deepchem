"""
Integration tests for singletask vector feature models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import unittest
import tempfile
import shutil
import tensorflow as tf
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor


class TestAPI(unittest.TestCase):
  """
  Test top-level API for ML models.
  """

  def test_singletask_sklearn_rf_ECFP_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "example.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    regression_metrics = [
        dc.metrics.Metric(dc.metrics.r2_score),
        dc.metrics.Metric(dc.metrics.mean_squared_error),
        dc.metrics.Metric(dc.metrics.mean_absolute_error)
    ]

    sklearn_model = RandomForestRegressor()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    _ = model.evaluate(train_dataset, regression_metrics, transformers)
    _ = model.evaluate(test_dataset, regression_metrics, transformers)

  def test_singletask_sklearn_rf_user_specified_regression_API(self):
    """Test of singletask RF USF regression API."""
    splittype = "specified"
    featurizer = dc.feat.UserDefinedFeaturizer(
        ["user-specified1", "user-specified2"])
    tasks = ["log-solubility"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "user_specified_example.csv")
    loader = dc.data.UserCSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.SpecifiedSplitter(input_file, "split")
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    regression_metrics = [
        dc.metrics.Metric(dc.metrics.r2_score),
        dc.metrics.Metric(dc.metrics.mean_squared_error),
        dc.metrics.Metric(dc.metrics.mean_absolute_error)
    ]

    sklearn_model = RandomForestRegressor()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train/test
    _ = model.evaluate(train_dataset, regression_metrics, transformers)
    _ = model.evaluate(test_dataset, regression_metrics, transformers)

  def test_singletask_sklearn_rf_RDKIT_descriptor_regression_API(self):
    """Test of singletask RF RDKIT-descriptor regression API."""
    splittype = "scaffold"
    featurizer = dc.feat.RDKitDescriptors()
    tasks = ["log-solubility"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "example.csv")
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [
        dc.trans.NormalizationTransformer(
            transform_X=True, dataset=train_dataset),
        dc.trans.ClippingTransformer(transform_X=True, dataset=train_dataset),
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    regression_metrics = [
        dc.metrics.Metric(dc.metrics.r2_score),
        dc.metrics.Metric(dc.metrics.mean_squared_error),
        dc.metrics.Metric(dc.metrics.mean_absolute_error)
    ]

    sklearn_model = RandomForestRegressor()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train/test
    _ = model.evaluate(train_dataset, regression_metrics, transformers)
    _ = model.evaluate(test_dataset, regression_metrics, transformers)

  def test_singletask_tf_mlp_ECFP_classification_API(self):
    """Test of Tensorflow singletask deepchem classification API."""
    n_features = 1024
    featurizer = dc.feat.CircularFingerprint(size=n_features)

    tasks = ["outcome"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "example_classification.csv")

    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]

    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    classification_metrics = [
        dc.metrics.Metric(dc.metrics.roc_auc_score),
        dc.metrics.Metric(dc.metrics.matthews_corrcoef),
        dc.metrics.Metric(dc.metrics.recall_score),
        dc.metrics.Metric(dc.metrics.accuracy_score)
    ]

    model = dc.models.TensorflowMultiTaskClassifier(len(tasks), n_features)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train/test
    _ = model.evaluate(train_dataset, classification_metrics, transformers)
    _ = model.evaluate(test_dataset, classification_metrics, transformers)

  def test_singletask_tg_mlp_ECFP_classification_API(self):
    """Test of TensorGraph singletask deepchem classification API."""
    n_features = 1024
    featurizer = dc.feat.CircularFingerprint(size=n_features)

    tasks = ["outcome"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "example_classification.csv")

    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]

    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    classification_metrics = [
        dc.metrics.Metric(dc.metrics.roc_auc_score),
        dc.metrics.Metric(dc.metrics.matthews_corrcoef),
        dc.metrics.Metric(dc.metrics.recall_score),
        dc.metrics.Metric(dc.metrics.accuracy_score)
    ]

    model = dc.models.TensorGraphMultiTaskClassifier(len(tasks), n_features)

    # Test Parameter getting and setting
    param, value = 'weight_decay_penalty_type', 'l2'
    assert model.get_params()[param] is None
    model.set_params(**{param: value})
    assert model.get_params()[param] == value

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train/test
    _ = model.evaluate(train_dataset, classification_metrics, transformers)
    _ = model.evaluate(test_dataset, classification_metrics, transformers)
