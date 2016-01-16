"""
Integration tests for singletask vector feature models. 
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
import numpy as np
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    self.input_file = os.path.join(current_dir, "example.csv")
    self.tasks = ["log-solubility"]
    self.smiles_field="smiles"
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
    """Straightforward test of deepchem API."""
    splittype = "random"
    feature_types = ["ECFP"]
    output_transforms = ["normalize"]
    input_transforms = []
    task_type = "regression"
    model_params = {}
    model_name = "rf_regressor"

    # Featurize input
    featurizer = DataFeaturizer(tasks=self.tasks,
                                smiles_field=self.smiles_field,
                                verbose=True)
    feature_file = os.path.join(self.feature_dir, "out.joblib")
    featurizer.featurize(self.input_file, ["ECFP"], feature_file)

    # Transform data into arrays for ML
    samples = FeaturizedSamples(self.samplesdir, [feature_file], reload=False)

    # Split into train/test
    train_samples, test_samples = samples.train_test_split(splittype,
      self.train_dir, self.test_dir)
    train_dataset = Dataset(self.train_dir, train_samples, feature_types)
    test_dataset = Dataset(self.test_dir, test_samples, feature_types)

    # Transforming train/test data
    train_dataset.transform(input_transforms, output_transforms)
    test_dataset.transform(input_transforms, output_transforms)

    # Fit model
    task_types = {task: task_type for task in self.tasks}
    model = Model.model_builder(model_name, task_types, model_params)
    model.fit(train_dataset)
    model.save(self.model_dir)

    # Eval model on train
    evaluator = Evaluator(model, test_dataset, verbose=True)
    with tempfile.NamedTemporaryFile() as test_csv_out:
      with tempfile.NamedTemporaryFile() as test_stats_out:
        evaluator.compute_model_performance(test_csv_out, test_stats_out)
