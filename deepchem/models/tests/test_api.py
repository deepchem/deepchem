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
from deepchem.featurizers.grid_featurizer import GridFeaturizer
from deepchem.datasets import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import LogTransformer
from deepchem.transformers import ClippingTransformer
from deepchem.models.test import TestAPI
from deepchem import metrics
from deepchem.metrics import Metric
from sklearn.ensemble import RandomForestRegressor
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier

class TestModelAPI(TestAPI):
  """
  Test top-level API for ML models.
  """
  def test_singletask_sklearn_rf_ECFP_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformers = []
    output_transformers = [NormalizationTransformer]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "example.csv"
    train_dataset, test_dataset, _, transformers, = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks)
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
    split_field = "split"
    compound_featurizers = []
    complex_featurizers = []
    input_transformers = []
    output_transformers = [NormalizationTransformer]
    model_params = {}
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "user_specified_example.csv"
    user_specified_features = ["user-specified1", "user-specified2"]
    train_dataset, test_dataset, _, transformers, = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks,
        user_specified_features=user_specified_features,
        split_field=split_field)
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
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformers = []
    output_transformers = [NormalizationTransformer]
    model_params = {}
    tasks = ["label"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    input_file = "../../../datasets/pdbbind_core_df.pkl.gz"
    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks,
        shard_size=50)
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
    compound_featurizers = [RDKitDescriptors()]
    complex_featurizers = []
    input_transformers = [NormalizationTransformer, ClippingTransformer]
    output_transformers = [NormalizationTransformer]
    tasks = ["log-solubility"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    model_params = {}
    input_file = "example.csv"
    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks)
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

  def test_singletask_keras_mlp_USF_regression_API(self):
    """Test of singletask MLP User Specified Features regression API."""
    from deepchem.models.keras_models.fcnet import SingleTaskDNN
    splittype = "scaffold"
    compound_featurizers = []
    complex_featurizers = []
    input_transformers = [NormalizationTransformer, ClippingTransformer]
    output_transformers = [NormalizationTransformer]
    feature_types = ["user_specified_features"]
    user_specified_features = ["evals"]
    tasks = ["u0"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2, "init": "glorot_uniform",
                    "nb_layers": 1, "batchnorm": False}

    input_file = "gbd3k.pkl.gz"
    protein_pdb_field = None
    ligand_pdb_field = None
    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers,
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks,
        protein_pdb_field=protein_pdb_field,
        ligand_pdb_field=ligand_pdb_field,
        user_specified_features=user_specified_features)
    model_params["data_shape"] = train_dataset.get_data_shape()
    regression_metrics = [Metric(metrics.r2_score),
                          Metric(metrics.mean_squared_error),
                          Metric(metrics.mean_absolute_error)]

    model = SingleTaskDNN(tasks, task_types, model_params, self.model_dir)

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
    from deepchem.models.keras_models.fcnet import MultiTaskDNN
    splittype = "scaffold"
    output_transformers = []
    input_transformers = []
    task_type = "classification"
    # TODO(rbharath): There should be some automatic check to ensure that all
    # required model_params are specified.
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2, "init": "glorot_uniform",
                    "nb_layers": 1, "batchnorm": False}

    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
             "task7", "task8", "task9", "task10", "task11", "task12",
             "task13", "task14", "task15", "task16"]
    task_types = {task: task_type for task in tasks}

    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []

    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks)
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

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    _ = evaluator.compute_model_performance(classification_metrics)

  '''
  # TODO(rbharath): This fails on many systems with an Illegal Instruction
  # error. Need to debug.
  from deepchem.featurizers.nnscore import NNScoreComplexFeaturizer
  def test_singletask_keras_mlp_NNScore_regression_API(self):
    """Test of singletask MLP NNScore regression API."""
    from deepchem.models.keras_models.fcnet import SingleTaskDNN
    splittype = "scaffold"
    compound_featurizers = []
    complex_featurizers = [NNScoreComplexFeaturizer()]
    input_transformers = [NormalizationTransformer, ClippingTransformer]
    output_transformers = [NormalizationTransformer]
    tasks = ["label"]
    task_type = "regression"
    task_types = {task: task_type for task in tasks}
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2, "init": "glorot_uniform",
                    "nb_layers": 1, "batchnorm": False}

    input_file = "nnscore_example.pkl.gz"
    protein_pdb_field = "protein_pdb"
    ligand_pdb_field = "ligand_pdb"
    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, tasks,
        protein_pdb_field=protein_pdb_field,
        ligand_pdb_field=ligand_pdb_field)
    model_params["data_shape"] = train_dataset.get_data_shape()
    
    model = SingleTaskDNN(task_types, model_params)
    self._create_model(train_dataset, test_dataset, model, transformers)

  #TODO(enf/rbharath): This should be uncommented now that 3D CNNs are in
  #                    keras. Need to upgrade the base version of keras for
  #                    deepchem.
  def test_singletask_cnn_GridFeaturizer_regression_API(self):
    """Test of singletask 3D ConvNet regression API."""
    splittype = "scaffold"
    compound_featurizers = []
    complex_featurizers = [GridFeaturizer(feature_types=["voxel_combined"],
                                          voxel_feature_types=["ecfp", "splif", "hbond",
                                                               "pi_stack", "cation_pi",
                                                               "salt_bridge", "charge"],
                                          voxel_width=0.5)]
    input_transforms = ["normalize", "truncate"]
    output_transforms = ["normalize"]
    task_type = "regression"
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2, "init": "glorot_uniform",
                    "loss_function": "mean_squared_error"}
    model_name = "convolutional_3D_regressor"
    protein_pdb_field = "protein_pdb"
    ligand_pdb_field = "ligand_pdb"
    input_file = "nnscore_example.pkl.gz"
    tasks = ["label"]
    self._create_model(splittype, compound_featurizers, complex_featurizers, input_transforms,
                       output_transforms, task_type, model_params, model_name,
                       input_file=input_file,
                       protein_pdb_field=protein_pdb_field,
                       ligand_pdb_field=ligand_pdb_field,
                       tasks=tasks)
    '''
