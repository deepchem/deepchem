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
from deepchem.utils.dataset import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
from deepchem.models.keras_models.fcnet import SingleTaskDNN
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.model_config import ModelConfig
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import LogTransformer
from deepchem.transformers import ClippingTransformer
from sklearn.ensemble import RandomForestRegressor

class TestAPI(unittest.TestCase):
  """
  Test top-level API for ML models."
  """
  def setUp(self):
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.smiles_field = "smiles"
    self.feature_dir = tempfile.mkdtemp()
    self.samples_dir = tempfile.mkdtemp()
    self.train_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()
    self.model_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.feature_dir)
    shutil.rmtree(self.samples_dir)
    shutil.rmtree(self.train_dir)
    shutil.rmtree(self.test_dir)
    shutil.rmtree(self.model_dir)

  def _create_model(self, train_dataset, test_dataset, model, transformers):
    """Helper method to create model for test."""

    # Fit model
    model.fit(train_dataset)
    model.save(self.model_dir)

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbose=True)
    with tempfile.NamedTemporaryFile() as train_csv_out:
      with tempfile.NamedTemporaryFile() as train_stats_out:
        _, _ = evaluator.compute_model_performance(
            train_csv_out, train_stats_out)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbose=True)
    with tempfile.NamedTemporaryFile() as test_csv_out:
      with tempfile.NamedTemporaryFile() as test_stats_out:
        _, _ = evaluator.compute_model_performance(
            test_csv_out, test_stats_out)

  def _featurize_train_test_split(self, splittype, compound_featurizers, 
                                  complex_featurizers,
                                  input_transformer_classes,
                                  output_transformer_classes, input_file, tasks, 
                                  protein_pdb_field=None, ligand_pdb_field=None,
                                  user_specified_features=None,
                                  split_field=None,
                                  shard_size=100):
    # Featurize input
    featurizers = compound_featurizers + complex_featurizers

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                protein_pdb_field=protein_pdb_field,
                                ligand_pdb_field=ligand_pdb_field,
                                compound_featurizers=compound_featurizers,
                                complex_featurizers=complex_featurizers,
                                user_specified_features=user_specified_features,
                                split_field=split_field,
                                verbose=True)
    

    #Featurizes samples and transforms them into NumPy arrays suitable for ML.
    #returns an instance of class FeaturizedSamples()

    samples = featurizer.featurize(input_file, self.feature_dir, self.samples_dir,
                                   shard_size=shard_size)

    # Splits featurized samples into train/test
    train_samples, test_samples = samples.train_test_split(
        splittype, self.train_dir, self.test_dir)

    use_user_specified_features = (user_specified_features is not None)
    train_dataset = Dataset(data_dir=self.train_dir, samples=train_samples, 
                            featurizers=featurizers, tasks=tasks,
                            use_user_specified_features=use_user_specified_features)
    test_dataset = Dataset(data_dir=self.test_dir, samples=test_samples, 
                           featurizers=featurizers, tasks=tasks,
                           use_user_specified_features=use_user_specified_features)

    # Initialize transformers
    input_transformers = []
    for transform_class in input_transformer_classes:
      input_transformers.append(transform_class(
          transform_X=True, dataset=train_dataset))
    output_transformers = []
    for transform_class in output_transformer_classes:
      output_transformers.append(transform_class(
          transform_y=True, dataset=train_dataset))
    transformers = input_transformers + output_transformers

    # Transforming train data
    for transformer in transformers:
      transformer.transform(train_dataset)
    # Transforming test data
    for transformer in transformers:
      transformer.transform(test_dataset)

    return train_dataset, test_dataset, input_transformers, output_transformers

  def test_singletask_sklearn_rf_ECFP_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformers = []
    output_transformers = [NormalizationTransformer]
    model_params = {}
    task_types = {"log-solubility": "regression"}
    input_file = "example.csv"
    train_dataset, test_dataset, _, transformers, = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, task_types.keys())
    model_params["data_shape"] = train_dataset.get_data_shape()

    model = SklearnModel(task_types, model_params, model_instance=RandomForestRegressor())
    self._create_model(train_dataset, test_dataset, model, transformers)

  def test_singletask_sklearn_rf_user_specified_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "specified"
    split_field = "split"
    compound_featurizers = []
    complex_featurizers = []
    input_transformers = []
    output_transformers = [NormalizationTransformer]
    model_params = {}
    task_types = {"log-solubility": "regression"}
    input_file = "user_specified_example.csv"
    user_specified_features = ["user-specified1", "user-specified2"]
    train_dataset, test_dataset, _, transformers, = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, task_types.keys(),
        user_specified_features=user_specified_features,
        split_field=split_field)
    model_params["data_shape"] = train_dataset.get_data_shape()

    model = SklearnModel(task_types, model_params, model_instance=RandomForestRegressor())
    self._create_model(train_dataset, test_dataset, model, transformers)

  def test_singletask_sklearn_rf_ECFP_regression_sharded_API(self):
    """Test of singletask RF ECFP regression API: sharded edition."""
    splittype = "scaffold"
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformers = []
    output_transformers = [NormalizationTransformer]
    model_params = {}
    task_types = {"label": "regression"}
    input_file = "../../../datasets/pdbbind_core_df.pkl.gz"
    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, task_types.keys(),
        shard_size=50)
    # We set shard size above to force the creation of multiple shards of the data.
    # pdbbind_core has ~200 examples.

    model_params["data_shape"] = train_dataset.get_data_shape()

    model = SklearnModel(task_types, model_params, model_instance=RandomForestRegressor())
    self._create_model(train_dataset, test_dataset, model, transformers)

  def test_singletask_sklearn_rf_RDKIT_descriptor_regression_API(self):
    """Test of singletask RF RDKIT-descriptor regression API."""
    splittype = "scaffold"
    compound_featurizers = [RDKitDescriptors()]
    complex_featurizers = []
    input_transformers = [NormalizationTransformer, ClippingTransformer]
    output_transformers = [NormalizationTransformer]
    task_types = {"log-solubility": "regression"}
    model_params = {}
    input_file = "example.csv"
    train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
        splittype, compound_featurizers, 
        complex_featurizers, input_transformers,
        output_transformers, input_file, task_types.keys())
    model_params["data_shape"] = train_dataset.get_data_shape()

    model = SklearnModel(task_types, model_params, model_instance=RandomForestRegressor())
    self._create_model(train_dataset, test_dataset, model, transformers)

  def test_singletask_keras_mlp_NNScore_regression_API(self):
    """Test of singletask MLP NNScore regression API."""
    splittype = "scaffold"
    compound_featurizers = []
    complex_featurizers = [NNScoreComplexFeaturizer()]
    input_transformers = [NormalizationTransformer, ClippingTransformer]
    output_transformers = [NormalizationTransformer]
    task_types = {"label": "regression"}
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
        output_transformers, input_file, task_types.keys(),
        protein_pdb_field=protein_pdb_field,
        ligand_pdb_field=ligand_pdb_field)
    model_params["data_shape"] = train_dataset.get_data_shape()
    
    model = SingleTaskDNN(task_types, model_params)
    self._create_model(train_dataset, test_dataset, model, transformers)


  def test_singletask_keras_mlp_USF_regression_API(self):
    """Test of singletask MLP User Specified Features regression API."""
    splittype = "scaffold"
    compound_featurizers = []
    complex_featurizers = []
    input_transformers = [NormalizationTransformer, ClippingTransformer]
    output_transformers = [NormalizationTransformer]
    feature_types = ["user_specified_features"]
    user_specified_features = ["evals"]
    task_types = {"u0": "regression"}
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
        output_transformers, input_file, task_types.keys(),
        protein_pdb_field=protein_pdb_field,
        ligand_pdb_field=ligand_pdb_field,
        user_specified_features=user_specified_features)
    model_params["data_shape"] = train_dataset.get_data_shape()

    model = SingleTaskDNN(task_types, model_params)
    self._create_model(train_dataset, test_dataset, model, transformers)


    #TODO(enf/rbharath): 3D CNN's are broken and must be fixed.
    '''
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

  def test_multitask_keras_mlp_ECFP_classification_API(self):
    """Straightforward test of Keras multitask deepchem classification API."""
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
        output_transformers, input_file, task_types.keys())
    model_params["data_shape"] = train_dataset.get_data_shape()
    
    model = MultiTaskDNN(task_types, model_params)
    self._create_model(train_dataset, test_dataset, model, transformers)

  def test_singletask_tf_mlp_ECFP_classificatioN_API(self):
    """Straightforward test of Tensorflow singletask deepchem classification API."""
    splittype = "scaffold"
    output_transformers = []
    input_transformers = []
    task_type = "classification"

    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []


    model_params = {}
    task_types = {"log-solubility": "regression"}
    input_file = "example.csv"
    input_transformers = []
    output_transformers = [NormalizationTransformer]

    #train_dataset, test_dataset, _, transformers = self._featurize_train_test_split(
    #    splittype, compound_featurizers, 
    #    complex_featurizers, input_transformers,
    #    output_transformers, input_file, task_types.keys())
    #model_params["data_shape"] = train_dataset.get_data_shape()
    config = ModelConfig()
    config.AddParam("batch_size", 32, "allowed")
    config.AddParam("num_classification_tasks", 1, "allowed")
    train = True
    logdir = self.model_dir
    model = TensorflowModel(task_types, model_params, config, train, logdir)
    #self._create_model(train_dataset, test_dataset, model, transformers)
