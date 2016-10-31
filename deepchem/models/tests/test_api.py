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
#from deepchem.featurizers import UserDefinedFeaturizer 
#from deepchem.featurizers.featurize import DataLoader
#from deepchem.featurizers.fingerprints import CircularFingerprint
#from deepchem.featurizers.basic import RDKitDescriptors
#from deepchem.datasets import Dataset
#from deepchem.utils.evaluate import Evaluator
#from deepchem.models import Model
#from deepchem.models.sklearn_models import SklearnModel
#from deepchem.transformers import NormalizationTransformer
#from deepchem.transformers import LogTransformer
#from deepchem.transformers import ClippingTransformer
#from deepchem.models.tests import TestAPI
#from deepchem import metrics
#from deepchem.metrics import Metric
#from sklearn.ensemble import RandomForestRegressor
#from deepchem.models.tensorflow_models import TensorflowModel
#from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
#from deepchem.splits import ScaffoldSplitter
#from deepchem.splits import SpecifiedSplitter
#from deepchem.models.keras_models.fcnet import MultiTaskDNN
#from deepchem.models.keras_models import KerasModel 


class TestAPI(unittest.TestCase):
  """
  Test top-level API for ML models.
  """
  def setUp(self):
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.smiles_field = "smiles"
    self.base_dir = tempfile.mkdtemp()
    self.data_dir = tempfile.mkdtemp()
    self.train_dir = tempfile.mkdtemp()
    self.valid_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()
    self.model_dir = tempfile.mkdtemp()
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

  def tearDown(self):
    shutil.rmtree(self.base_dir)
    shutil.rmtree(self.data_dir)
    shutil.rmtree(self.train_dir)
    shutil.rmtree(self.valid_dir)
    shutil.rmtree(self.test_dir)
    # TODO(rbharath): Removing this causes crashes for some reason. Need to
    # debug.
    #shutil.rmtree(self.model_dir)

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
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, self.train_dir, self.test_dir)

    transformers = [dc.transformers.NormalizationTransformer(
        transform_y=True, dataset=train_dataset)]
    regression_metrics = [dc.metrics.Metric(metrics.r2_score),
                          dc.metrics.Metric(metrics.mean_squared_error),
                          dc.metrics.Metric(metrics.mean_absolute_error)]

    sklearn_model = RandomForestRegressor()
    model = dc.models.SklearnModel(sklearn_model)

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    _ = model.evaluate(train_dataset, regression_metrics, transformers)
    _ = model.evaluate(valid_dataset, regression_metrics, transformers)

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
        dataset = transformer.transform(dataset)

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
        dataset = transformer.transform(dataset)

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
        dataset = transformer.transform(dataset)

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
