"""
Integration tests for singletask vector feature models.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
import tempfile
import shutil
from deepchem.utils.featurize import DataFeaturizer
from deepchem.utils.featurize import FeaturizedSamples
from deepchem.utils.dataset import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
# We need to import models so they can be created by model_builder
import deepchem.models.deep
import deepchem.models.standard
import deepchem.models.deep3d

class TestSingletaskVectorAPI(unittest.TestCase):
  """
  Test top-level API for singletask vector models."
  """
  def setUp(self):
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.smiles_field = "smiles"
    self.feature_dir = tempfile.mkdtemp()
    self.samplesdir = tempfile.mkdtemp()
    self.train_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()
    self.model_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.feature_dir)
    shutil.rmtree(self.samplesdir)
    shutil.rmtree(self.train_dir)
    shutil.rmtree(self.test_dir)
    shutil.rmtree(self.model_dir)

  def _create_model(self, splittype, feature_types, input_transforms,
                    output_transforms, task_type, model_params, model_name,
                    input_file, tasks, protein_pdb_field=None, ligand_pdb_field=None):
    """Helper method to create model for test."""
    # Featurize input
    input_file = os.path.join(self.current_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                protein_pdb_field=protein_pdb_field,
                                ligand_pdb_field=ligand_pdb_field,
                                verbose=True)
    feature_file = os.path.join(self.feature_dir, "out.joblib")
    featurizer.featurize(input_file, feature_types, feature_file)

    # Transform data into arrays for ML
    samples = FeaturizedSamples(self.samplesdir, [feature_file],
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
    task_types = {task: task_type for task in tasks}
    model_params["data_shape"] = train_dataset.get_data_shape()
    model = Model.model_builder(model_name, task_types, model_params)
    model.fit(train_dataset)
    model.save(self.model_dir)

    # Eval model on train
    evaluator = Evaluator(model, test_dataset, verbose=True)
    with tempfile.NamedTemporaryFile() as test_csv_out:
      with tempfile.NamedTemporaryFile() as test_stats_out:
        _, _ = evaluator.compute_model_performance(
            test_csv_out, test_stats_out)

  def test_singletask_rf_ECFP_regression_API(self):
    """Test of singletask RF ECFP regression API."""
    splittype = "scaffold"
    feature_types = ["ECFP"]
    input_transforms = []
    output_transforms = ["normalize"]
    task_type = "regression"
    model_params = {"batch_size": 5}
    model_name = "rf_regressor"
    input_file = "example.csv"
    tasks = ["log-solubility"]
    self._create_model(splittype, feature_types, input_transforms,
                       output_transforms, task_type, model_params, model_name,
                       input_file=input_file, tasks=tasks)


  def test_singletask_rf_RDKIT_descriptor_regression_API(self):
    """Test of singletask RF RDKIT-descriptor regression API."""
    splittype = "scaffold"
    feature_types = ["RDKIT-descriptors"]
    input_transforms = ["normalize", "truncate"]
    output_transforms = ["normalize"]
    task_type = "regression"
    model_params = {"batch_size": 5}
    model_name = "rf_regressor"
    input_file = "example.csv"
    tasks = ["log-solubility"]
    self._create_model(splittype, feature_types, input_transforms,
                       output_transforms, task_type, model_params, model_name,
                       input_file=input_file, tasks=tasks)

  def test_singletask_mlp_NNScore_regression_API(self):
    """Test of singletask MLP NNScore regression API."""
    splittype = "scaffold"
    feature_types = ["NNScore"]
    input_transforms = ["normalize", "truncate"]
    output_transforms = ["normalize"]
    task_type = "regression"
    model_params = {"nb_hidden": 10, "activation": "relu",
                    "dropout": .5, "learning_rate": .01,
                    "momentum": .9, "nesterov": False,
                    "decay": 1e-4, "batch_size": 5,
                    "nb_epoch": 2}
    model_name = "singletask_deep_regressor"
    protein_pdb_field = "protein_pdb"
    ligand_pdb_field = "ligand_pdb"
    input_file = "nnscore_example.pkl.gz"
    tasks = ["label"]
    self._create_model(splittype, feature_types, input_transforms,
                       output_transforms, task_type, model_params, model_name,
                       input_file=input_file,
                       protein_pdb_field=protein_pdb_field,
                       ligand_pdb_field=ligand_pdb_field,
                       tasks=tasks)

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
    self.samplesdir = tempfile.mkdtemp()
    self.train_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()
    self.model_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.feature_dir)
    shutil.rmtree(self.samplesdir)
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
    feature_file = os.path.join(self.feature_dir, "out.joblib")
    featurizer.featurize(self.input_file, feature_types, feature_file)

    # Transform data into arrays for ML
    samples = FeaturizedSamples(self.samplesdir, [feature_file],
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
