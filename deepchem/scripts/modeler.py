"""
Top level script to featurize input, train models, and evaluate them.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import argparse
import glob
import os
import multiprocessing as mp
from functools import partial
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.featurize import FeaturizedSamples
from deepchem.utils.dataset import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
# We need to import models so they can be created by model_builder
import deepchem.models.deep
import deepchem.models.standard
import deepchem.models.deep3d

def add_featurize_group(featurize_cmd):
  """Adds flags for featurizization."""
  featurize_group = featurize_cmd.add_argument_group("Input Specifications")
  featurize_group.add_argument(
      "--input-files", required=1, nargs="+",
      help="Input file with data.")
  featurize_group.add_argument(
      "--user-specified-features", type=str, nargs="+",
      help="Optional field that holds pre-computed feature vector")
  featurize_group.add_argument(
      "--tasks", type=str, nargs="+", required=1,
      help="Name of measured field to predict.")
  featurize_group.add_argument(
      "--split-field", type=str, default=None,
      help="Name of field specifying train/test split.")
  featurize_group.add_argument(
      "--smiles-field", type=str, default="smiles",
      help="Name of field specifying SMILES for molecule.")
  featurize_group.add_argument(
      "--id-field", type=str, default=None,
      help="Name of field specifying unique identifier for molecule.\n"
           "If none is specified, then smiles-field is used as identifier.")
  featurize_group.add_argument(
      "--threshold", type=float, default=None,
      help="If specified, will be used to binarize real-valued target-fields.")
  featurize_group.add_argument(
      "--protein-pdb-field", type=str, default=None,
      help="Name of field holding protein pdb.")
  featurize_group.add_argument(
      "--ligand-pdb-field", type=str, default=None,
      help="Name of field holding ligand pdb.")
  featurize_group.add_argument(
      "--ligand-mol2-field", type=str, default=None,
      help="Name of field holding ligand mol2.")

def add_transforms_group(cmd):
  """Adds flags for data transforms."""
  transform_group = cmd.add_argument_group("Transform Group")
  transform_group.add_argument(
      "--input-transforms", nargs="+", default=[],
      choices=["normalize", "truncate", "log"],
      help="Transforms to apply to input data.")
  transform_group.add_argument(
      "--output-transforms", nargs="+", default=[],
      choices=["normalize", "log"],
      help="Supported transforms are 'log' and 'normalize'. 'None' will be taken\n"
           "to mean no transforms are required.")
  transform_group.add_argument(
      "--feature-types", nargs="+", required=1,
      choices=["user-specified-features", "ECFP", "RDKIT-descriptors", "NNScore"],
      help="Featurizations of data to use.\n"
           "'features' denotes user-defined features.\n"
           "'fingerprints' denotes ECFP fingeprints.\n"
           "'descriptors' denotes RDKit chem descriptors.\n")
  transform_group.add_argument(
      "--splittype", type=str, default="scaffold",
      choices=["scaffold", "random", "specified"],
      help="Type of train/test data-split. 'scaffold' uses Bemis-Murcko scaffolds.\n"
           "specified requires that split be in original data.")
  transform_group.add_argument(
      "--weight-positives", type=bool, default=False,
      help="Weight positive examples to have same total weight as negatives.")

def add_model_group(fit_cmd):
  """Adds flags for specifying models."""
  group = fit_cmd.add_argument_group("model")
  group.add_argument(
      "--model", required=1,
      choices=["logistic", "rf_classifier", "rf_regressor",
               "linear", "ridge", "lasso", "lasso_lars", "elastic_net",
               "singletask_deep_classifier", "multitask_deep_classifier",
               "singletask_deep_regressor", "multitask_deep_regressor",
               "convolutional_3D_regressor"],
      help="Type of model to build. Some models may allow for\n"
           "further specification of hyperparameters. See flags below.")

  group = fit_cmd.add_argument_group("Neural Net Parameters")
  group.add_argument(
      "--nb-hidden", type=int, default=500,
      help="Number of hidden neurons for NN models.")
  group.add_argument(
      "--learning-rate", type=float, default=0.01,
      help="Learning rate for NN models.")
  group.add_argument(
      "--dropout", type=float, default=0.5,
      help="Learning rate for NN models.")
  group.add_argument(
      "--nb-epoch", type=int, default=50,
      help="Number of epochs for NN models.")
  group.add_argument(
      "--batch-size", type=int, default=32,
      help="Number of examples per minibatch for NN models.")
  group.add_argument(
      "--loss-function", type=str, default="mean_squared_error",
      help="Loss function type.")
  group.add_argument(
      "--decay", type=float, default=1e-4,
      help="Learning rate decay for NN models.")
  group.add_argument(
      "--activation", type=str, default="relu",
      help="NN activation function.")
  group.add_argument(
      "--momentum", type=float, default=.9,
      help="Momentum for stochastic gradient descent.")
  group.add_argument(
      "--nesterov", action="store_true",
      help="If set, use Nesterov acceleration.")


def add_model_command(subparsers):
  """Adds flags for model subcommand."""
  model_cmd = subparsers.add_parser(
      "model", help="Combines featurize, train-test-split, fit, eval into one\n"
      "command for user convenience.")
  model_cmd.add_argument(
      "--featurize", action="store_true",
      help="Perform the featurization step.")
  model_cmd.add_argument(
      "--generate-dataset", action="store_true",
      help="Generate dataset from featurized data.")
  model_cmd.add_argument(
      "--train-test-split", action="store_true",
      help="Perform the train-test-split step.")
  model_cmd.add_argument(
      "--fit", action="store_true",
      help="Perform model fit step.")
  model_cmd.add_argument(
      "--eval", action="store_true",
      help="Perform model eval step.")
  model_cmd.add_argument(
      "--eval-full", action="store_true",
      help="Evaluate model on full dataset.")
  model_cmd.add_argument(
      "--base-dir", type=str, default=None,
      help="The base directory for the model.")
  model_cmd.add_argument(
      "--feature-dir", type=str, default=None,
      help="The feature storage directory for the model.")
  model_cmd.add_argument(
      "--data-dir", type=str, default=None,
      help="The data storage directory for the model.")
  model_cmd.add_argument(
      "--model-dir", type=str, default=None,
      help="The model storage directory for the model.")
  add_featurize_group(model_cmd)

  add_transforms_group(model_cmd)
  add_model_group(model_cmd)
  model_cmd.set_defaults(func=create_model)

def extract_model_params(args):
  """
  Given input arguments, return a dict specifiying model parameters.
  """
  params = ["nb_hidden", "learning_rate", "dropout",
            "nb_epoch", "decay", "batch_size", "loss_function",
            "activation", "momentum", "nesterov"]

  model_params = {param : getattr(args, param) for param in params}
  return model_params

def ensure_exists(dirs):
  """Creates dirs if they don't exist."""
  for directory in dirs:
    if not os.path.exists(directory):
      os.makedirs(directory)

def create_model(args):
  """Creates a model"""
  model_name = args.model
  if args.base_dir is not None:
    feature_dir = os.path.join(args.base_dir, "features")
    data_dir = os.path.join(args.base_dir, "data")
    model_dir = os.path.join(args.base_dir, "model")
    ensure_exists([args.base_dir, feature_dir, data_dir, model_dir])
  else:
    if (args.model_dir is None or
        args.data_dir is None or
        args.feature_dir is None):
      raise ValueError("If base-dir not specified, must specify "
                       "feature-dir, data-dir, model-dir.")

    feature_dir, model_dir, data_dir = (args.feature_dir, args.model_dir,
                                        args.data_dir)
    ensure_exists([feature_dir, data_dir, model_dir])

  if args.featurize:
    print("+++++++++++++++++++++++++++++++++")
    print("Perform featurization")
    featurize_inputs(
        feature_dir, data_dir, args.input_files, args.user_specified_features,
        args.tasks, args.smiles_field, args.split_field, args.id_field,
        args.threshold, args.protein_pdb_field,
        args.ligand_pdb_field, args.ligand_mol2_field)

  if args.generate_dataset:
    print("+++++++++++++++++++++++++++++++++")
    print("Generate dataset for featurized samples")
    samples_dir = os.path.join(data_dir, "samples")
    samples = FeaturizedSamples(samples_dir, reload_data=True)

    print("Generating dataset.")
    full_data_dir = os.path.join(data_dir, "full-data")
    full_dataset = Dataset(full_data_dir, samples, args.feature_types)

    print("Transform data.")
    full_dataset.transform(args.input_transforms, args.output_transforms)


  if args.train_test_split:
    print("+++++++++++++++++++++++++++++++++")
    print("Perform train-test split")
    train_test_split(
        args.input_transforms, args.output_transforms, args.feature_types,
        args.splittype, data_dir)

  if args.fit:
    print("+++++++++++++++++++++++++++++++++")
    print("Fit model")
    model_params = extract_model_params(args)
    fit_model(
        model_name, model_params, model_dir, data_dir)

  if args.eval:
    print("+++++++++++++++++++++++++++++++++")
    print("Eval Model on Train")
    print("-------------------")
    train_dir = os.path.join(data_dir, "train-data")
    csv_out_train = os.path.join(data_dir, "train.csv")
    stats_out_train = os.path.join(data_dir, "train-stats.txt")
    eval_trained_model(
        model_name, model_dir, train_dir, csv_out_train,
        stats_out_train)

    print("Eval Model on Test")
    print("------------------")
    test_dir = os.path.join(data_dir, "test-data")
    csv_out_test = os.path.join(data_dir, "test.csv")
    stats_out_test = os.path.join(data_dir, "test-stats.txt")
    eval_trained_model(
        model_name, model_dir, test_dir, csv_out_test,
        stats_out_test)

  if args.eval_full:
    print("+++++++++++++++++++++++++++++++++")
    print("Eval Model on Full Dataset")
    print("--------------------------")
    full_data_dir = os.path.join(data_dir, "full-data")
    csv_out_full = os.path.join(data_dir, "full.csv")
    stats_out_full = os.path.join(data_dir, "full-stats.txt")
    eval_trained_model(
        model_name, model_dir, full_data_dir, csv_out_full,
        stats_out_full)

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Modes')
  add_model_command(subparsers)
  return parser.parse_args(input_args)

def shard_inputs(input_file):
  input_file_no_ext = os.path.splitext(input_file)


def featurize_inputs(feature_dir, data_dir, input_files,
                     user_specified_features, tasks, smiles_field,
                     split_field, id_field, threshold, protein_pdb_field, 
                     ligand_pdb_field, ligand_mol2_field):

  """Allows for parallel data featurization."""
  featurize_input_partial = partial(featurize_input,
                                    feature_dir=feature_dir,
                                    user_specified_features=user_specified_features,
                                    tasks=tasks,
                                    smiles_field=smiles_field,
                                    split_field=split_field,
                                    id_field=id_field,
                                    threshold=threshold,
                                    protein_pdb_field=protein_pdb_field,
                                    ligand_pdb_field=ligand_pdb_field,
                                    ligand_mol2_field=ligand_mol2_field)

  for input_file in input_files:
    featurize_input_partial(input_file)

  dataset_files = glob.glob(os.path.join(feature_dir, "*.joblib"))

  print("Writing samples to disk.")
  samples_dir = os.path.join(data_dir, "samples")
  FeaturizedSamples(samples_dir, dataset_files)

def featurize_input(input_file, feature_dir, user_specified_features, tasks,
                    smiles_field, split_field, id_field, threshold, protein_pdb_field,
                     ligand_pdb_field, ligand_mol2_field):
  """Featurizes raw input data."""
  featurizer = DataFeaturizer(tasks=tasks,
                              smiles_field=smiles_field,
                              split_field=split_field,
                              id_field=id_field,
                              threshold=threshold,
                              protein_pdb_field=protein_pdb_field,
                              ligand_pdb_field=ligand_pdb_field,
                              ligand_mol2_field=ligand_mol2_field,
                              user_specified_features=user_specified_features,
                              verbose=True)

  featurizer.featurize(input_file, FeaturizedSamples.feature_types, feature_dir)

def train_test_split(input_transforms, output_transforms,
                     feature_types, splittype, data_dir):
  """Saves transformed model."""

  samples_dir = os.path.join(data_dir, "samples")
  samples = FeaturizedSamples(samples_dir, reload_data=True)

  print("Split data into train/test")
  train_samples_dir = os.path.join(data_dir, "train-samples")
  test_samples_dir = os.path.join(data_dir, "test-samples")
  train_samples, test_samples = samples.train_test_split(
      splittype, train_samples_dir, test_samples_dir)

  train_data_dir = os.path.join(data_dir, "train-data")
  test_data_dir = os.path.join(data_dir, "test-data")

  print("Generating train dataset.")
  train_dataset = Dataset(train_data_dir, train_samples, feature_types)

  print("Generating test dataset.")
  test_dataset = Dataset(test_data_dir, test_samples, feature_types)

  print("Transforming train data.")
  train_dataset.transform(input_transforms, output_transforms)

  print("Transforming test data.")
  test_dataset.transform(input_transforms, output_transforms)

def fit_model(model_name, model_params, model_dir, data_dir):
  """Builds model from featurized data."""
  task_type = Model.get_task_type(model_name)
  train_dir = os.path.join(data_dir, "train-data")
  train = Dataset(train_dir)

  task_types = {task: task_type for task in train.get_task_names()}
  model_params["data_shape"] = train.get_data_shape()

  model = Model.model_builder(model_name, task_types, model_params)
  model.fit(train)
  model.save(model_dir)

def eval_trained_model(model_type, model_dir, data_dir,
                       csv_out, stats_out):
  """Evaluates a trained model on specified data."""
  model = Model.load(model_type, model_dir)
  data = Dataset(data_dir)

  evaluator = Evaluator(model, data, verbose=True)
  _, perf_df = evaluator.compute_model_performance(csv_out, stats_out)
  print("Model Performance.")
  print(perf_df)

def main():
  """Invokes argument parser."""
  args = parse_args()
  args.func(args)

if __name__ == "__main__":
  main()
