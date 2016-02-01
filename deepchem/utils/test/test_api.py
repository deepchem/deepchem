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
# We need to import models so they can be created by model_builder
import deepchem.models.deep
import deepchem.models.standard
import deepchem.models.deep3d
from deepchem.models.deep import SingleTaskDNN
from deepchem.models.deep import MultiTaskDNN
from deepchem.models.standard import SklearnModel

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

  def _create_model(self, train_dataset, test_dataset, model):
    """Helper method to create model for test."""

    # Fit model

    model.fit(train_dataset)
    model.save(self.model_dir)

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, verbose=True)
    with tempfile.NamedTemporaryFile() as train_csv_out:
      with tempfile.NamedTemporaryFile() as train_stats_out:
        _, _ = evaluator.compute_model_performance(
            train_csv_out, train_stats_out)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, verbose=True)
    with tempfile.NamedTemporaryFile() as test_csv_out:
      with tempfile.NamedTemporaryFile() as test_stats_out:
        _, _ = evaluator.compute_model_performance(
            test_csv_out, test_stats_out)

  def _featurize_train_test_split(self, splittype, compound_featurizers, complex_featurizers, input_transforms,
                        output_transforms, input_file, tasks, protein_pdb_field=None, ligand_pdb_field=None):

    # Featurize input
    featurizers = compound_featurizers + complex_featurizers

    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                protein_pdb_field=protein_pdb_field,
                                ligand_pdb_field=ligand_pdb_field,
                                compound_featurizers=compound_featurizers,
                                complex_featurizers=complex_featurizers,
                                verbose=True)

    #Featurizes samples and transforms them into NumPy arrays suitable for ML.
    #returns an instance of class FeaturizedSamples()

    samples = featurizer.featurize(input_file, self.feature_dir, self.samples_dir)

    # Splits featurized samples into train/test
    train_samples, test_samples = samples.train_test_split(
        splittype, self.train_dir, self.test_dir)

    train_dataset = Dataset(data_dir=self.train_dir, samples=train_samples, 
                            featurizers=featurizers, tasks=tasks)
    test_dataset = Dataset(data_dir=self.test_dir, samples=test_samples, 
                           featurizers=featurizers, tasks=tasks)

    # Transforming train/test data
    train_dataset.transform(input_transforms, output_transforms)
    test_dataset.transform(input_transforms, output_transforms)

    return train_dataset, test_dataset

  def test_singletask_rf_ECFP_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transforms = []
    output_transforms = ["normalize"]
    model_params = {"batch_size": 5}
    task_types = {"log-solubility": "regression"}
    input_file = "example.csv"
    train_dataset, test_dataset = self._featurize_train_test_split(splittype, compound_featurizers, 
                                                    complex_featurizers, input_transforms,
                                                    output_transforms, input_file, task_types.keys())
    model_params["data_shape"] = train_dataset.get_data_shape()

    from sklearn.ensemble import RandomForestRegressor
    model = SklearnModel(task_types, model_params, model_instance=RandomForestRegressor())
    self._create_model(train_dataset, test_dataset, model)


  def test_singletask_rf_RDKIT_descriptor_regression_API(self):
    """Test of singletask RF RDKIT-descriptor regression API."""
    splittype = "scaffold"
    compound_featurizers = [RDKitDescriptors()]
    complex_featurizers = []
    input_transforms = ["normalize", "truncate"]
    output_transforms = ["normalize"]
    task_types = {"log-solubility": "regression"}
    model_params = {"batch_size": 5}
    input_file = "example.csv"
    train_dataset, test_dataset = self._featurize_train_test_split(splittype, compound_featurizers, 
                                                    complex_featurizers, input_transforms,
                                                    output_transforms, input_file, task_types.keys())
    model_params["data_shape"] = train_dataset.get_data_shape()

    from sklearn.ensemble import RandomForestRegressor
    model = SklearnModel(task_types, model_params, model_instance=RandomForestRegressor())
    self._create_model(train_dataset, test_dataset, model)

  def test_singletask_mlp_NNScore_regression_API(self):
    """Test of singletask MLP NNScore regression API."""
    splittype = "scaffold"
    compound_featurizers = []
    complex_featurizers = [NNScoreComplexFeaturizer()]
    input_transforms = ["normalize", "truncate"]
    output_transforms = ["normalize"]
    task_types = {"label": "regression"}
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2, "init": "glorot_uniform"}

    input_file = "nnscore_example.pkl.gz"
    protein_pdb_field = "protein_pdb"
    ligand_pdb_field = "ligand_pdb"
    train_dataset, test_dataset = self._featurize_train_test_split(splittype, compound_featurizers, 
                                                    complex_featurizers, input_transforms,
                                                    output_transforms, input_file, task_types.keys(),
                                                    protein_pdb_field=protein_pdb_field,
                                                    ligand_pdb_field=ligand_pdb_field)
    model_params["data_shape"] = train_dataset.get_data_shape()
    
    model = SingleTaskDNN(task_types, model_params)
    self._create_model(train_dataset, test_dataset, model)


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

  def test_multitask_mlp_ECFP_classification_API(self):
    """Straightforward test of multitask deepchem classification API."""
    splittype = "scaffold"
    output_transforms = []
    input_transforms = []
    task_type = "classification"
    # TODO(rbharath): There should be some automatic check to ensure that all
    # required model_params are specified.
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2, "init": "glorot_uniform"}

    input_file = os.path.join(self.current_dir, "multitask_example.csv")
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
                  "task7", "task8", "task9", "task10", "task11", "task12",
                  "task13", "task14", "task15", "task16"]
    task_types = {task: task_type for task in tasks}

    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []

    train_dataset, test_dataset = self._featurize_train_test_split(splittype, compound_featurizers, 
                                                    complex_featurizers, input_transforms,
                                                    output_transforms, input_file, task_types.keys())
    model_params["data_shape"] = train_dataset.get_data_shape()
    


    model = MultiTaskDNN(task_types, model_params)
    self._create_model(train_dataset, test_dataset, model)


'''
class TestMultitaskVectorAPI(unittest.TestCase):
  """
  Test top-level API for singletask vector models."
  """
  def setUp(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.input_file = os.path.join(current_dir, "multitask_example.csv")
    self.tasks = ["task0", "task1", "task2", "task3", "task4", "task5", "task6",
                  "task7", "task8", "task9", "task10", "task11", "task12",
                  "task13", "task14", "task15", "task16"]
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

  def test_API(self):
    """Straightforward test of multitask deepchem classification API."""
    splittype = "scaffold"
    feature_types = ["ECFP"]
    output_transforms = []
    input_transforms = []
    task_type = "classification"
    # TODO(rbharath): There should be some automatic check to ensure that all
    # required model_params are specified.
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2}
    model_name = "multitask_deep_classifier"

    # Featurize input
    featurizer = DataFeaturizer(tasks=self.tasks,
                                smiles_field=self.smiles_field,
                                verbose=True)
    feature_files = featurizer.featurize(self.input_file, feature_types, self.feature_dir)

    # Transform data into arrays for ML
    samples = FeaturizedSamples(self.samples_dir, feature_files,
                                reload_data=False)

    # Split into train/test
    train_samples, test_samples = samples.train_test_split(
        splittype, self.train_dir, self.test_dir)
    train_dataset = Dataset(self.train_dir, train_samples, feature_types)
    test_dataset = Dataset(self.test_dir, test_samples, feature_types)

    # Transforming train/test data
    train_dataset.transform(input_transforms, output_transforms)
    test_dataset.transform(input_transforms, output_transforms)

    # Fit model
    task_types = {task: task_type for task in self.tasks}
    model_params["data_shape"] = train_dataset.get_data_shape()
    model = Model.model_builder(model_name, task_types, model_params)
    model.fit(train_dataset)
    model.save(self.model_dir)

    # Eval model on train
    evaluator = Evaluator(model, test_dataset, verbose=True)
    with tempfile.NamedTemporaryFile() as test_csv_out:
      with tempfile.NamedTemporaryFile() as test_stats_out:
        evaluator.compute_model_performance(test_csv_out, test_stats_out)
'''