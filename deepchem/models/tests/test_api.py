"""
Integration tests for singletask vector feature models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import unittest
import tempfile
import shutil
import tensorflow as tf
from keras import backend as K
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor

class TestAPI(unittest.TestCase):
  """
  Test top-level API for ML models.
  """
  def test_singletask_sklearn_rf_ECFP_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    featurizer = dc.featurizers.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    input_file = os.path.join(self.current_dir, "example.csv")
    loader = dc.loaders.DataLoader(
        tasks=tasks, smiles_field=self.smiles_field,
        featurizer=featurizer, verbosity="low")
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [dc.transformers.NormalizationTransformer(
        transform_y=True, dataset=train_dataset)]
    regression_metrics = [dc.metrics.Metric(dc.metrics.r2_score),
                          dc.metrics.Metric(dc.metrics.mean_squared_error),
                          dc.metrics.Metric(dc.metrics.mean_absolute_error)]

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
    featurizer = dc.featurizers.UserDefinedFeaturizer(
        ["user-specified1", "user-specified2"])
    tasks = ["log-solubility"]
    input_file = os.path.join(self.current_dir, "user_specified_example.csv")
    loader = dc.loaders.DataLoader(
        tasks=tasks, smiles_field=self.smiles_field, featurizer=featurizer,
        verbosity="low")
    dataset = loader.featurize(input_file, debug=True)

    splitter = dc.splits.SpecifiedSplitter(input_file, "split")
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [dc.transformers.NormalizationTransformer(
        transform_y=True, dataset=train_dataset)]
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    regression_metrics = [dc.metrics.Metric(dc.metrics.r2_score),
                          dc.metrics.Metric(dc.metrics.mean_squared_error),
                          dc.metrics.Metric(dc.metrics.mean_absolute_error)]

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
    featurizer = dc.featurizers.RDKitDescriptors()
    tasks = ["log-solubility"]

    input_file = os.path.join(self.current_dir, "example.csv")
    loader = dc.loaders.DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)

    transformers = [
        dc.transformers.NormalizationTransformer(
            transform_X=True, dataset=train_dataset),
        dc.transformers.ClippingTransformer(
            transform_X=True, dataset=train_dataset),
        dc.transformers.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)]
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    regression_metrics = [dc.metrics.Metric(dc.metrics.r2_score),
                          dc.metrics.Metric(dc.metrics.mean_squared_error),
                          dc.metrics.Metric(dc.metrics.mean_absolute_error)]

    sklearn_model = RandomForestRegressor()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train/test
    _ = model.evaluate(train_dataset, regression_metrics, transformers)
    _ = model.evaluate(test_dataset, regression_metrics, transformers)

  def test_multitask_keras_mlp_ECFP_classification_API(self):
    """Test of Keras multitask deepchem classification API."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      task_type = "classification"
      input_file = os.path.join(self.current_dir, "multitask_example.csv")
      tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
               "task7", "task8", "task9", "task10", "task11", "task12",
               "task13", "task14", "task15", "task16"]

      n_features = 1024
      featurizer = dc.featurizers.CircularFingerprint(size=n_features)
      loader = dc.loaders.DataLoader(
          tasks=tasks, smiles_field=self.smiles_field,
          featurizer=featurizer, verbosity="low")
      dataset = loader.featurize(input_file)

      splitter = dc.splits.ScaffoldSplitter()
      train_dataset, test_dataset = splitter.train_test_split(dataset)

      metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score),
                 dc.metrics.Metric(dc.metrics.matthews_corrcoef),
                 dc.metrics.Metric(dc.metrics.recall_score),
                 dc.metrics.Metric(dc.metrics.accuracy_score)]
      
      keras_model = dc.models.MultiTaskDNN(
          len(tasks), n_features, "classification", dropout=0.)
      model = dc.models.KerasModel(keras_model)

      # Fit trained model
      model.fit(train_dataset)
      model.save()

      # Eval model on train/test
      _ = model.evaluate(train_dataset, metrics)
      _ = model.evaluate(test_dataset, metrics)

  def test_singletask_tf_mlp_ECFP_classification_API(self):
    """Test of Tensorflow singletask deepchem classification API."""
    n_features = 1024
    featurizer = dc.featurizers.CircularFingerprint(size=n_features)

    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")

    loader = dc.loaders.DataLoader(
        tasks=tasks, smiles_field=self.smiles_field,
        featurizer=featurizer, verbosity="low")
    dataset = loader.featurize(input_file)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)
    
    transformers = [dc.transformers.NormalizationTransformer(
        transform_y=True, dataset=train_dataset)]

    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        dataset = transformer.transform(dataset)

    classification_metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score),
                              dc.metrics.Metric(dc.metrics.matthews_corrcoef),
                              dc.metrics.Metric(dc.metrics.recall_score),
                              dc.metrics.Metric(dc.metrics.accuracy_score)]

    tensorflow_model = dc.models.TensorflowMultiTaskClassifier(
        len(tasks), n_features)
    model = dc.models.TensorflowModel(tensorflow_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train/test
    _ = model.evaluate(train_dataset, classification_metrics, transformers)
    _ = model.evaluate(test_dataset, classification_metrics, transformers)
