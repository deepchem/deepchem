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
    model_params = {}
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
    model_params["data_shape"] = train_dataset.get_data_shape()
    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="regression",
                         model_instance=RandomForestRegressor())

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
    model_params = {}
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

    model_params["data_shape"] = train_dataset.get_data_shape()
    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="regression",
                         model_instance=RandomForestRegressor())

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
    model_params = {}
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
    model_params["data_shape"] = train_dataset.get_data_shape()
    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="regression",
                         model_instance=RandomForestRegressor())

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
    model_params = {}
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

    model_params["data_shape"] = train_dataset.get_data_shape()
    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    model = SklearnModel(tasks, task_types, model_params, self.model_dir,
                         mode="regression",
                         model_instance=RandomForestRegressor())
  

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(regression_metrics)


  #### TODO(rbharath): This test is being disabled since deepchem no longer
  #### accepts this format of input. Decide whether this test should be deleted
  #### altogether or replaced.
  #def test_singletask_keras_mlp_USF_regression_API(self):
  #  """Test of singletask MLP User Specified Features regression API."""
  #  from deepchem.models.keras_models.fcnet import SingleTaskDNN
  #  featurizer = UserDefinedFeaturizer(["evals"])
  #  tasks = ["u0"]
  #  task_type = "regression"
  #  task_types = {task: task_type for task in tasks}
  #  model_params = {"nb_hidden": 10, "activation": "relu",
  #                  "dropout": .5, "learning_rate": .01,
  #                  "momentum": .9, "nesterov": False,
  #                  "decay": 1e-4, "batch_size": 5,
  #                  "nb_epoch": 2, "init": "glorot_uniform",
  #                  "nb_layers": 1, "batchnorm": False}

  #  input_file = os.path.join(self.current_dir, "gbd3k.pkl.gz")
  #  loader = DataLoader(tasks=tasks,
  #                      smiles_field=self.smiles_field,
  #                      featurizer=featurizer,
  #                      verbosity="low")
  #  dataset = loader.featurize(input_file, self.data_dir)

  #  splitter = ScaffoldSplitter()
  #  train_dataset, test_dataset = splitter.train_test_split(
  #      dataset, self.train_dir, self.test_dir)

  #  input_transformers = [
  #    NormalizationTransformer(transform_X=True, dataset=train_dataset),
  #    ClippingTransformer(transform_X=True, dataset=train_dataset)]
  #  output_transformers = [
  #    NormalizationTransformer(transform_y=True, dataset=train_dataset)]
  #  transformers = input_transformers + output_transformers

  #  for dataset in [train_dataset, test_dataset]:
  #    for transformer in transformers:
  #      transformer.transform(dataset)

  #  model_params["data_shape"] = train_dataset.get_data_shape()
  #  regression_metrics = [Metric(metrics.r2_score),
  #                        Metric(metrics.mean_squared_error),
  #                        Metric(metrics.mean_absolute_error)]

  #  # Fit trained model
  #  model.fit(train_dataset)
  #  model.save()

  #  # Eval model on train
  #  evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
  #  _ = evaluator.compute_model_performance(regression_metrics)

  #  # Eval model on test
  #  evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
  #  _ = evaluator.compute_model_performance(regression_metrics)


  def test_multitask_keras_mlp_ECFP_classification_API(self):
    """Straightforward test of Keras multitask deepchem classification API."""
    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    with g.as_default():
      task_type = "classification"
      # TODO(rbharath): There should be some automatic check to ensure that all
      # required model_params are specified.
      # TODO(rbharath): Turning off dropout to make tests behave.
      model_params = {"nb_hidden": 10, "activation": "relu",
                      "dropout": .0, "learning_rate": .01,
                      "momentum": .9, "nesterov": False,
                      "decay": 1e-4, "batch_size": 5,
                      "nb_epoch": 2, "init": "glorot_uniform",
                      "nb_layers": 1, "batchnorm": False}

      input_file = os.path.join(self.current_dir, "multitask_example.csv")
      tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
               "task7", "task8", "task9", "task10", "task11", "task12",
               "task13", "task14", "task15", "task16"]
      task_types = {task: task_type for task in tasks}

      featurizer = CircularFingerprint(size=1024)

      loader = DataLoader(tasks=tasks,
                          smiles_field=self.smiles_field,
                          featurizer=featurizer,
                          verbosity="low")
      dataset = loader.featurize(input_file, self.data_dir)
      splitter = ScaffoldSplitter()
      train_dataset, test_dataset = splitter.train_test_split(
          dataset, self.train_dir, self.test_dir)

      transformers = []
      model_params["data_shape"] = train_dataset.get_data_shape()
      classification_metrics = [Metric(metrics.roc_auc_score),
                                Metric(metrics.matthews_corrcoef),
                                Metric(metrics.recall_score),
                                Metric(metrics.accuracy_score)]
      
      model = MultiTaskDNN(tasks, task_types, model_params, self.model_dir)

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
    splittype = "scaffold"
    output_transformers = []
    input_transformers = []
    task_type = "classification"

    featurizer = CircularFingerprint(size=1024)

    tasks = ["outcome"]
    task_type = "classification"
    task_types = {task: task_type for task in tasks}
    input_file = os.path.join(self.current_dir, "example_classification.csv")

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

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)
