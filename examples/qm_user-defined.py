"""
Example script for using user-defined features and QM data
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
import tempfile
import shutil
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.featurize import FeaturizedSamples
from deepchem.utils.dataset import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model

# List of models available to imported into model_builder
#import deepchem.models.deep
#import deepchem.models.standard
#import deepchem.models.deep3d
from deepchem.models.deep import SingleTaskDNN
#from deepchem.models.deep import MultiTaskDNN
#from deepchem.models.standard import SklearnModel

def featurize_train_test_split(splittype, compound_featurizers, 
                                complex_featurizers, input_transforms,
                                output_transforms, input_file, tasks, 
                                feature_dir, samples_dir, train_dir, 
                                test_dir, smiles_field,
                                protein_pdb_field=None, ligand_pdb_field=None,
                                user_specified_features=None, shard_size=100):
  # Featurize input
  featurizers = compound_featurizers + complex_featurizers

  current_dir = os.path.dirname(os.path.abspath(__file__))
  input_file = os.path.join(current_dir, input_file)
  featurizer = DataFeaturizer(tasks=tasks,
                              smiles_field=smiles_field,
                              protein_pdb_field=protein_pdb_field,
                              ligand_pdb_field=ligand_pdb_field,
                              compound_featurizers=compound_featurizers,
                              complex_featurizers=complex_featurizers,
                              user_specified_features=user_specified_features,
                              verbose=True)
  

  #Featurizes samples and transforms them into NumPy arrays suitable for ML.
  #returns an instance of class FeaturizedSamples()

  samples = featurizer.featurize(input_file, feature_dir, samples_dir,
                                 shard_size=shard_size)

  # Splits featurized samples into train/test
  train_samples, test_samples = samples.train_test_split(
      splittype, train_dir, test_dir)

  use_user_specified_features = False
  if user_specified_features is not None:
    use_user_specified_features = True

  train_dataset = Dataset(data_dir=train_dir, samples=train_samples, 
                          featurizers=featurizers, tasks=tasks,
                          use_user_specified_features=use_user_specified_features)
  test_dataset = Dataset(data_dir=test_dir, samples=test_samples, 
                         featurizers=featurizers, tasks=tasks,
                         use_user_specified_features=use_user_specified_features)

  # Transforming train/test data
  train_dataset.transform(input_transforms, output_transforms)
  test_dataset.transform(input_transforms, output_transforms)

  return train_dataset, test_dataset

def create_and_eval_model(train_dataset, test_dataset, model, model_dir):

  # Fit model

  model.fit(train_dataset)
  model.save(model_dir)

  # Eval model on train
  evaluator = Evaluator(model, train_dataset, verbose=True)
  with tempfile.NamedTemporaryFile() as train_csv_out:
    with tempfile.NamedTemporaryFile() as train_stats_out:
      _, performance_df = evaluator.compute_model_performance(
          train_csv_out, train_stats_out)
  print("train_performance_df")
  print(performance_df)

  # Eval model on test
  evaluator = Evaluator(model, test_dataset, verbose=True)
  with tempfile.NamedTemporaryFile() as test_csv_out:
    with tempfile.NamedTemporaryFile() as test_stats_out:
      _, performance_df = evaluator.compute_model_performance(
          test_csv_out, test_stats_out)
  print("train_performance_df")
  print(performance_df)
  
def main():

  feature_dir = tempfile.mkdtemp()
  samples_dir = tempfile.mkdtemp()
  train_dir = tempfile.mkdtemp()
  test_dir = tempfile.mkdtemp()
  model_dir = tempfile.mkdtemp()

  splittype = "scaffold"
  compound_featurizers = []
  complex_featurizers = []
  input_transforms = ["normalize", "truncate"]
  output_transforms = ["normalize"]
  feature_types = ["user_specified_features"]
  user_specified_features = ["evals"]
  task_types = {"u0": "regression"}
  model_params = {"nb_hidden": 10, "activation": "relu",
                  "dropout": .5, "learning_rate": .01,
                  "momentum": .9, "nesterov": False,
                  "decay": 1e-4, "batch_size": 5,
                  "nb_epoch": 2, "init": "glorot_uniform",
                  "nb_layers": 1, "batchnorm": False}

  input_file = "../datasets/gbd3k.pkl.gz"
  smiles_field = "smiles"
  protein_pdb_field = None
  ligand_pdb_field = None
  train_dataset, test_dataset = featurize_train_test_split(splittype, 
                                    compound_featurizers, complex_featurizers, 
                                    input_transforms, output_transforms, input_file, 
                                    task_types.keys(), feature_dir, samples_dir,
                                    train_dir, test_dir, smiles_field, 
                                    protein_pdb_field=protein_pdb_field,
                                    ligand_pdb_field=ligand_pdb_field,
                                    user_specified_features=user_specified_features)

  model_params["data_shape"] = train_dataset.get_data_shape()
  model = SingleTaskDNN(task_types, model_params)
  create_and_eval_model(train_dataset, test_dataset, model, model_dir)

  shutil.rmtree(feature_dir)
  shutil.rmtree(samples_dir)
  shutil.rmtree(train_dir)
  shutil.rmtree(test_dir)
  shutil.rmtree(model_dir)

if __name__ == "__main__":
  main()
