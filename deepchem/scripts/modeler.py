"""
Top level script to featurize input, train models, and evaluate them.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import argparse
import glob
import os
from functools import partial
from deepchem.utils.featurize import DataFeaturizer
from deepchem.utils.featurize import FeaturizedSamples
from deepchem.utils.dataset import Dataset
from deepchem.utils.evaluate import Evaluator
from deepchem.models import Model
# We need to import models so they can be created by model_builder
import deepchem.models.deep
import deepchem.models.standard
import deepchem.models.deep3d

# TODO(rbharath): Are any commands except for create_model actually used? Due to
# the --skip-foo flags, it's possible to run all functionality directly through
# create_model. Perhaps trim the fat and delete the remaining commands.

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
      "--feature-dir", type=str, required=0,
      help="Directory where featurized dataset will be stored.\n"
           "Will be created if does not exist")
  featurize_group.add_argument(
      "--parallel", type=float, default=None,
      help="Use multiprocessing will be used to parallelize featurization.")

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
      "--mode", default="singletask",
      choices=["singletask", "multitask"],
      help="Type of model being built.")
  transform_group.add_argument(
      "--feature-types", nargs="+", required=1,
      choices=["user-specified-features", "ECFP", "RDKIT-descriptors"],
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
      "--skip-featurization", action="store_true",
      help="If set, skip the featurization step.")
  model_cmd.add_argument(
      "--skip-train-test-split", action="store_true",
      help="If set, skip the train-test-split step.")
  model_cmd.add_argument(
      "--skip-fit", action="store_true",
      help="If set, skip model fit step.")
  model_cmd.add_argument(
      "--skip-eval", action="store_true",
      help="If set, skip model eval step.")
  model_cmd.add_argument(
      "--base-dir", type=str, required=1,
      help="The base directory for the model.")
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
  return(model_params)

def ensure_exists(dirs):
  for directory in dirs:
    if not os.path.exists(directory):
      os.makedirs(directory)

def create_model(args):
  """Creates a model"""
  base_dir = args.base_dir
  feature_dir = os.path.join(base_dir, "features")
  data_dir = os.path.join(base_dir, "data")
  model_dir = os.path.join(base_dir, "model")
  ensure_exists([base_dir, feature_dir, data_dir, model_dir])

  model_name = args.model

  print("+++++++++++++++++++++++++++++++++")
  print("Perform featurization")
  if not args.skip_featurization:
    featurize_inputs(
        feature_dir, args.input_files,
        args.user_specified_features, args.tasks,
        args.smiles_field, args.split_field, args.id_field, args.threshold,
        args.parallel)

  print("+++++++++++++++++++++++++++++++++")
  print("Perform train-test split")
  paths = [feature_dir]
  if not args.skip_train_test_split:
    train_test_split(
        paths, args.input_transforms, args.output_transforms, args.feature_types,
        args.splittype, args.mode, data_dir)

  print("+++++++++++++++++++++++++++++++++")
  print("Fit model")
  if not args.skip_fit:
    model_params = extract_model_params(args)
    fit_model(
        model_name, model_params, model_dir, data_dir)

  print("+++++++++++++++++++++++++++++++++")
  print("Eval Model on Train")
  print("-------------------")
  if not args.skip_eval:
    csv_out_train = os.path.join(data_dir, "train.csv")
    stats_out_train = os.path.join(data_dir, "train-stats.txt")
    csv_out_test = os.path.join(data_dir, "test.csv")
    stats_out_test = os.path.join(data_dir, "test-stats.txt")
    train_dir = os.path.join(data_dir, "train-data")
    eval_trained_model(
        model_name, model_dir, train_dir, csv_out_train,
        stats_out_train, split="train")
  print("Eval Model on Test")
  print("------------------")
  if not args.skip_eval:
    test_dir = os.path.join(data_dir, "test-data")
    eval_trained_model(
        model_name, model_dir, test_dir, csv_out_test,
        stats_out_test, split="test")

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Modes')
  add_model_command(subparsers)
  return parser.parse_args(input_args)

def featurize_inputs(feature_dir, input_files,
                     user_specified_features, tasks, smiles_field,
                     split_field, id_field, threshold, parallel):

  featurize_input_partial = partial(featurize_input,
                                    feature_dir=feature_dir,
                                    user_specified_features=user_specified_features,
                                    tasks=tasks,
                                    smiles_field=smiles_field,
                                    split_field=split_field,
                                    id_field=id_field,
                                    threshold=threshold)

  if parallel:
    pool = mp.Pool(int(mp.cpu_count()/2))
    pool.map(featurize_input_partial, input_files)
    pool.terminate()
  else:
    for input_file in input_files:
      featurize_input_partial(input_file)

def featurize_input(input_file, feature_dir, user_specified_features, tasks,
                    smiles_field, split_field, id_field, threshold):
  """Featurizes raw input data."""
  featurizer = DataFeaturizer(tasks=tasks,
                              smiles_field=smiles_field,
                              split_field=split_field,
                              id_field=id_field,
                              threshold=threshold,
                              user_specified_features=user_specified_features,
                              verbose=True)
  out = os.path.join(
      feature_dir, "%s.joblib" %(os.path.splitext(os.path.basename(input_file))[0]))
  featurizer.featurize(input_file, FeaturizedSamples.feature_types, out)

def train_test_split(paths, input_transforms, output_transforms,
                     feature_types, splittype, mode, data_dir):
  """Saves transformed model."""

  dataset_files = []
  for path in paths:
    dataset_files += glob.glob(os.path.join(path, "*.joblib"))
  print("paths")
  print(paths)

  print("Loading featurized data.")
  samples_dir = os.path.join(data_dir, "samples")
  samples = FeaturizedSamples(samples_dir, dataset_files, reload=False)
  
  print("Split data into train/test")
  train_samples_dir = os.path.join(data_dir, "train-samples")
  test_samples_dir = os.path.join(data_dir, "test-samples")
  train_samples, test_samples = samples.train_test_split(splittype,
    train_samples_dir, test_samples_dir)

  train_data_dir = os.path.join(data_dir, "train-data")
  test_data_dir = os.path.join(data_dir, "test-data")

  print("Generating train data.")
  train_dataset = Dataset(train_data_dir, train_samples, feature_types)
  print("Generating test data.")
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
                       csv_out, stats_out, split="test"):
  """Evaluates a trained model on specified data."""
  model = Model.load(model_type, model_dir)
  print("eval_trained_model()")
  print("data_dir")
  print(data_dir)
  
  data = Dataset(data_dir)

  evaluator = Evaluator(model, data, verbose=True)
  evaluator.compute_model_performance(csv_out, stats_out)

def main():
  """Invokes argument parser."""
  args = parse_args()
  args.func(args)

if __name__ == "__main__":
  main()
