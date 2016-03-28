"""
Tests for dataset creation
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import unittest
import tempfile
import os
import shutil
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import NormalizationTransformer

class TestAPI(unittest.TestCase):
  """
  Test top-level API for ML models."
  """
  def setUp(self):
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    self.test_data_dir = os.path.join(self.current_dir, "../../models/test")
    self.smiles_field = "smiles"
    self.feature_dir = tempfile.mkdtemp()
    self.samples_dir = tempfile.mkdtemp()
    self.data_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.feature_dir)
    shutil.rmtree(self.samples_dir)
    shutil.rmtree(self.data_dir)

  # TODO(rbharath): There should be a more natural way to create a dataset
  # object, perhaps just starting from (Xs, ys, ws)
  def _create_dataset(self, compound_featurizers, complex_featurizers,
                      input_transformer_classes, output_transformer_classes,
                      input_file, tasks,
                      protein_pdb_field=None, ligand_pdb_field=None,
                      user_specified_features=None,
                      split_field=None,
                      shard_size=100):
    # Featurize input
    featurizers = compound_featurizers + complex_featurizers

    input_file = os.path.join(self.test_data_dir, input_file)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field=self.smiles_field,
                                protein_pdb_field=protein_pdb_field,
                                ligand_pdb_field=ligand_pdb_field,
                                compound_featurizers=compound_featurizers,
                                complex_featurizers=complex_featurizers,
                                user_specified_features=user_specified_features,
                                split_field=split_field,
                                verbosity="low")

    samples = featurizer.featurize(input_file, self.feature_dir, self.samples_dir,
                                   shard_size=shard_size)
    use_user_specified_features = (user_specified_features is not None)
    dataset = Dataset(data_dir=self.data_dir, samples=samples, 
                      featurizers=featurizers, tasks=tasks,
                      use_user_specified_features=use_user_specified_features)
    return dataset

  def _load_solubility_data(self):
    """Loads solubility data from example.csv"""
    compound_featurizers = [CircularFingerprint(size=1024)]
    complex_featurizers = []
    input_transformer_classes = []
    output_transformer_classes = [NormalizationTransformer]
    task_types = {"log-solubility": "regression"}
    input_file = "example.csv"
    return self._create_dataset(
        compound_featurizers, complex_featurizers,
        input_transformer_classes, output_transformer_classes,
        input_file, task_types.keys())

  def test_get_task_names(self):
    """Test that get_task_names returns correct task_names"""
    solubility_dataset = self._load_solubility_data()
    assert solubility_dataset.get_task_names() == ["log-solubility"]

  def test_get_data_shape(self):
    """Test that get_data_shape returns currect data shape"""
    solubility_dataset = self._load_solubility_data()
    assert solubility_dataset.get_data_shape() == (1024,) 

  def test_len(self):
    """Test that len(dataset) works."""
    solubility_dataset = self._load_solubility_data()
    assert len(solubility_dataset) == 10
  
  def test_iterbatches(self):
    """Test that iterating over batches of data works."""
    solubility_dataset = self._load_solubility_data()
    batch_size = 2
    data_shape = solubility_dataset.get_data_shape()
    tasks = solubility_dataset.get_task_names()
    for (X_b, y_b, w_b, ids_b)  in solubility_dataset.iterbatches(batch_size):
      assert X_b.shape == (batch_size,) + data_shape
      assert y_b.shape == (batch_size,) + (len(tasks),)
      assert w_b.shape == (batch_size,) + (len(tasks),)
      assert ids_b.shape == (batch_size,)
