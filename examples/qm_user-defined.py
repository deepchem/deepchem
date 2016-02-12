"""
Hyperparameter search script for QM data.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Evan Feinberg, and Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import tempfile
import shutil
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.featurize import FeaturizedSamples
from deepchem.utils.dataset import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
# We need to import models so they can be created by model_builder
#import deepchem.models.deep
import itertools
import pandas as pd
from sklearn.externals import joblib

def user_featurize_and_split(input_file, feature_dir, samples_dir, train_dir,
                        test_dir, splittype, feature_types,
                        user_specified_features, input_transforms,
                        output_transforms, tasks, feature_files):
  """Featurize inputs with user-specified-features and do train-test split."""

  featurizer = DataFeaturizer(tasks=tasks,
                              smiles_field="smiles",
                              user_specified_features=user_specified_features,
                              verbose=True)
  
  print("About to featurize.")
  samples = featurizer.featurize(input_file, feature_dir,
                                       samples_dir, shard_size=8)
  print("Completed Featurization")
  train_samples, test_samples = samples.train_test_split(
      splittype, train_dir, test_dir)
  print("Finished train test split.")
  train_dataset = Dataset(train_dir, tasks, train_samples, feature_types,
                         use_user_specified_features=True)
  test_dataset = Dataset(test_dir, tasks, test_samples, feature_types,
                         use_user_specified_features=True)
  print("Finished creating train test datasets")
  # Transforming train/test data
  train_dataset.transform(input_transforms, output_transforms)
  test_dataset.transform(input_transforms, output_transforms)
  print("Finished Transforming train test data.")

  return train_dataset, test_dataset

def create_and_eval_model(train_dataset, test_dataset, task_type,
                          model_params, model_name, model_dir, tasks):
  """Helper method to create model for test."""
  # Fit model
  task_types = {task: task_type for task in tasks}
  model_params["data_shape"] = train_dataset.get_data_shape()
  print("Creating Model object.")
  import deepchem.models.deep
  model = Model.model_builder(model_name, task_types, model_params)
  print("About to fit model")
  model.fit(train_dataset)
  print("Done fitting, about to save...")
  model.save(model_dir)

  # Eval model on train
  evaluator = Evaluator(model, train_dataset, verbose=True)
  with tempfile.NamedTemporaryFile() as train_csv_out:
    with tempfile.NamedTemporaryFile() as train_stats_out:
      _, performance_df = evaluator.compute_model_performance(
          train_csv_out, train_stats_out)
  print("train_performance_df")
  print(performance_df)   

  evaluator = Evaluator(model, test_dataset, verbose=True)
  with tempfile.NamedTemporaryFile() as test_csv_out:
    with tempfile.NamedTemporaryFile() as test_stats_out:
      _, performance_df = evaluator.compute_model_performance(
          test_csv_out, test_stats_out)
  print("test_performance_df")
  print(performance_df)  

  return performance_df.iterrows().next()[1]["r2_score"]

def nnscore_hyperparam_search(train_dataset, test_dataset,
                              hyperparam_filename, tasks):
  """Test of singletask MLP NNScore regression API."""

  task_type = "regression"
  model_params = {"activation": "relu",
                  "dropout": 0.,
                  "momentum": .9, "nesterov": False,
                  "decay": 1e-4, "batch_size": 5,
                  "nb_epoch": 10}
  model_name = "singletask_deep_regressor"
  #model_name = "SingleTaskDNN"
  nb_hidden_vals = [10, 100]
  learning_rate_vals = [.01, .001]
  init_vals = ["glorot_uniform"]
  hyperparameters = [nb_hidden_vals, learning_rate_vals, init_vals]
  hyperparameter_rows = []
  for hyperparameter_tuple in itertools.product(*hyperparameters):
    nb_hidden, learning_rate, init = hyperparameter_tuple
    model_params["nb_hidden"] = nb_hidden
    model_params["learning_rate"] = learning_rate
    model_params["init"] = init

    model_dir = tempfile.mkdtemp()

    r2_score = create_and_eval_model(train_dataset, test_dataset, task_type,
                                     model_params, model_name, model_dir, tasks)

    print("%s: %s" % (hyperparameter_tuple, r2_score))
    hyperparameter_rows.append(list(hyperparameter_tuple) + [r2_score])

    shutil.rmtree(model_dir)

  hyperparameter_df = pd.DataFrame(hyperparameter_rows,
                                   columns=('nb_hidden', 'learning_rate',
                                            'init', 'r2_score'))
  hyperparameter_df.to_csv(hyperparam_filename)

def main():
  """Conduct hyperparameter search"""

  feature_dir = tempfile.mkdtemp()
  samples_dir = tempfile.mkdtemp()
  train_dir = tempfile.mkdtemp()
  test_dir = tempfile.mkdtemp()
  splittype = "scaffold"
  feature_types = ["user_specified_features"]
  user_specified_features = ["evals"]
  input_transforms = ["normalize", "truncate"]
  output_transforms = ["normalize"]
  input_file = "../datasets/gbd3k.pkl.gz"
  feature_files = ["None"]
  tasks = ["u0"]
  results_file = "qm_hyperparameter_results.csv"

  train_dataset, test_dataset = user_featurize_and_split(
      input_file, feature_dir, samples_dir, train_dir,
      test_dir, splittype, feature_types, 
      user_specified_features, input_transforms,
      output_transforms, tasks, feature_files)

  nnscore_hyperparam_search(train_dataset, test_dataset,
                            results_file, tasks)

  shutil.rmtree(feature_dir)
  shutil.rmtree(samples_dir)
  shutil.rmtree(train_dir)
  shutil.rmtree(test_dir)

if __name__ == "__main__":
  main()
