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
from deepchem.datasets import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import LogTransformer
from deepchem.transformers import ClippingTransformer
from deepchem.hyperparameters import HyperparamOpt
from sklearn.ensemble import RandomForestRegressor
from deepchem.splits import RandomSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import SpecifiedSplitter

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
    self.valid_dir = tempfile.mkdtemp()
    self.test_dir = tempfile.mkdtemp()
    self.model_dir = tempfile.mkdtemp()
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

  def tearDown(self):
    shutil.rmtree(self.feature_dir)
    shutil.rmtree(self.samples_dir)
    shutil.rmtree(self.train_dir)
    shutil.rmtree(self.valid_dir)
    shutil.rmtree(self.test_dir)
    # TODO(rbharath): Removing this causes crashes for some reason. Need to
    # debug.
    #shutil.rmtree(self.model_dir)

  def _hyperparam_opt(self, model_builder, params_dict, train_dataset,
                      valid_dataset, output_transformers, tasks, task_types, metric,
                      logdir=None):

    optimizer = HyperparamOpt(model_builder, tasks, task_types, verbosity="low")
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, output_transformers,
      metric, logdir=logdir)

  def _create_model(self, train_dataset, test_dataset, model, transformers,
                    metrics):
    """Helper method to create model for test."""

    # Fit trained model
    model.fit(train_dataset)
    model.save()

    # Eval model on train
    evaluator = Evaluator(model, train_dataset, transformers, verbosity=True)
    with tempfile.NamedTemporaryFile() as train_csv_out:
      with tempfile.NamedTemporaryFile() as train_stats_out:
        _ = evaluator.compute_model_performance(
            metrics, train_csv_out.name, train_stats_out)

    # Eval model on test
    evaluator = Evaluator(model, test_dataset, transformers, verbosity=True)
    with tempfile.NamedTemporaryFile() as test_csv_out:
      with tempfile.NamedTemporaryFile() as test_stats_out:
        _ = evaluator.compute_model_performance(
            metrics, test_csv_out.name, test_stats_out)

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
                                verbosity="low")
    

    #Featurizes samples and transforms them into NumPy arrays suitable for ML.
    #returns an instance of class FeaturizedSamples()

    samples = featurizer.featurize(input_file, self.feature_dir, self.samples_dir,
                                   shard_size=shard_size)

    # Splits featurized samples into train/test
    assert splittype in ["random", "specified", "scaffold"]
    if splittype == "random":
      splitter = RandomSplitter()
    elif splittype == "specified":
      splitter = SpecifiedSplitter()
    elif splittype == "scaffold":
      splitter = ScaffoldSplitter()
    train_samples, test_samples = splitter.train_test_split(
        samples, self.train_dir, self.test_dir)

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
