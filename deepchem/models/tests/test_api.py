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
from deepchem.featurizers import UserDefinedFeaturizer 
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.basic import RDKitDescriptors
from deepchem.featurizers.grid_featurizer import GridFeaturizer
from deepchem.datasets import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import LogTransformer
from deepchem.transformers import ClippingTransformer
from deepchem.models.tests import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
from sklearn.ensemble import RandomForestRegressor
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import SpecifiedSplitter
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.keras_models import KerasModel 
import tensorflow as tf
from keras import backend as K

class TestModelAPI(TestAPI):
  """
  Test top-level API for ML models.
  """
  def test_singletask_sklearn_rf_ECFP_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    featurizer = CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example.csv")
    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)

    input_transformers = []
    output_transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    transformers = input_transformers + output_transformers
    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    sklearn_model = RandomForestRegressor()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

  def test_singletask_sklearn_rf_user_specified_regression_API(self):
    """Test of singletask RF USF regression API."""
    splittype = "specified"
    featurizer = UserDefinedFeaturizer(["user-specified1", "user-specified2"])
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "user_specified_example.csv")
    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir, debug=True)

    splitter = SpecifiedSplitter(input_file, "split")
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)

    input_transformers = []
    output_transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    transformers = input_transformers + output_transformers
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        transformer.transform(dataset)

    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    sklearn_model = RandomForestRegressor()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

  def test_singletask_sklearn_rf_ECFP_regression_sharded_API(self):
    """Test of singletask RF ECFP regression API: sharded edition."""
    splittype = "scaffold"
    featurizer = CircularFingerprint(size=1024)
    tasks = ["label"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(
        self.current_dir, "../../../datasets/pdbbind_core_df.pkl.gz")

    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)
    input_transformers = []
    output_transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    transformers = input_transformers + output_transformers
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        transformer.transform(dataset)
    # We set shard size above to force the creation of multiple shards of the data.
    # pdbbind_core has ~200 examples.
    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    sklearn_model = RandomForestRegressor()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

  def test_singletask_sklearn_rf_RDKIT_descriptor_regression_API(self):
    """Test of singletask RF RDKIT-descriptor regression API."""
    splittype = "scaffold"
    featurizer = RDKitDescriptors()
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example.csv")
    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)

    input_transformers = [
        NormalizationTransformer(transform_X=True, dataset=train_dataset),
        ClippingTransformer(transform_X=True, dataset=train_dataset)]
    output_transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    transformers = input_transformers + output_transformers
    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        transformer.transform(dataset)

    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    sklearn_model = RandomForestRegressor()
    model = SklearnModel(sklearn_model, self.model_dir)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

  def test_multitask_keras_mlp_ECFP_classification_API(self):
    """Straightforward test of Keras multitask deepchem classification API."""
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
      featurizer = CircularFingerprint(size=n_features)
      loader = DataLoader(tasks=tasks,
                          smiles_field=self.smiles_field,
                          featurizer=featurizer,
                          verbosity="low")
      dataset = loader.featurize(input_file, self.data_dir)
      splitter = ScaffoldSplitter()
      train_dataset, test_dataset = splitter.train_test_split(
          dataset, self.train_dir, self.test_dir)

      transformers = []
      classification_metrics = [Metric(metrics.roc_auc_score),
                                Metric(metrics.matthews_corrcoef),
                                Metric(metrics.recall_score),
                                Metric(metrics.accuracy_score)]
      
      keras_model = MultiTaskDNN(len(tasks), n_features, "classification",
                                 dropout=0.)
      model = KerasModel(keras_model, self.model_dir)

      # Fit trained model
      model.fit(train_dataset)
      model.save()

      # Eval model on train
      evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
      _ = evaluator.compute_model_performance(classification_metrics)

      # Eval model on test
      evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
      _ = evaluator.compute_model_performance(classification_metrics)

  def test_singletask_tf_mlp_ECFP_classification_API(self):
    """Straightforward test of Tensorflow singletask deepchem classification API."""
    n_features = 1024
    featurizer = CircularFingerprint(size=n_features)

    tasks = ["outcome"]
    input_file = os.path.join(self.current_dir, "example_classification.csv")

    loader = DataLoader(tasks=tasks,
                        smiles_field=self.smiles_field,
                        featurizer=featurizer,
                        verbosity="low")
    dataset = loader.featurize(input_file, self.data_dir)

    splitter = ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)
    
    transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]

    for dataset in [train_dataset, test_dataset]:
      for transformer in transformers:
        transformer.transform(dataset)

    classification_metrics = [Metric(metrics.roc_auc_score),
                              Metric(metrics.matthews_corrcoef),
                              Metric(metrics.recall_score),
                              Metric(metrics.accuracy_score)]

    tensorflow_model = TensorflowMultiTaskClassifier(
        len(tasks), n_features, self.model_dir)
    model = TensorflowModel(tensorflow_model, self.model_dir)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)
