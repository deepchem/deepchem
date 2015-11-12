"""
Top level script to featurize input, train models, and evaluate them.
"""
import argparse
import numpy as np
from deep_chem.utils.featurize import generate_directories
from deep_chem.utils.featurize import extract_data
from deep_chem.utils.featurize import generate_targets
from deep_chem.utils.featurize import generate_features
from deep_chem.utils.featurize import generate_fingerprints
from deep_chem.utils.featurize import generate_descriptors
from deep_chem.models.deep import fit_singletask_mlp
from deep_chem.models.deep import fit_multitask_mlp
from deep_chem.models.deep3d import fit_3D_convolution
from deep_chem.models.standard import fit_singletask_models
from deep_chem.utils.load import get_target_names
from deep_chem.utils.load import process_datasets
from deep_chem.utils.evaluate import results_to_csv

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Modes')
 
  # FEATURIZE FLAGS
  featurize_cmd = subparsers.add_parser("featurize",
                      help="Featurize raw input data.")
  featurize_cmd.add_argument("--input-file", required=1,
                      help="Input file with data.")
  featurize_cmd.add_argument("--input-type", default="csv",
                      choices=["xlsx", "csv", "pandas", "sdf"],
                      help="Type of input file. If pandas, input must be a pkl.gz\n"
                           "containing a pandas dataframe. If sdf, should be in\n"
                           "(perhaps gzipped) sdf file.")
  featurize_cmd.add_argument("--delimiter", default=",",
                      help="If csv input, delimiter to use for read csv file")
  featurize_cmd.add_argument("--fields", required=1, nargs="+",
                      help = "Names of fields.")
  featurize_cmd.add_argument("--field-types", required=1, nargs="+",
                      choices=["string", "float", "list-string", "list-float", "ndarray"],
                      help="Type of data in fields.")
  featurize_cmd.add_argument("--feature-endpoints", type=str, nargs="+",
                      help="Optional endpoint that holds pre-computed feature vector")
  featurize_cmd.add_argument("--prediction-endpoint", type=str, required=1,
                      help="Name of measured endpoint to predict.")
  featurize_cmd.add_argument("--split-endpoint", type=str, default=None,
                      help="Name of endpoint specifying train/test split.")
  featurize_cmd.add_argument("--smiles-endpoint", type=str, default="smiles",
                      help="Name of endpoint specifying SMILES for molecule.")
  featurize_cmd.add_argument("--threshold", type=float, default=None,
                      help="If specified, will be used to binarize real-valued prediction-endpoint.")
  featurize_cmd.add_argument("--name", required=1,
                      help="Name of the dataset.")
  featurize_cmd.add_argument("--out", required=1,
                      help="Folder to generate processed dataset in.")
  featurize_cmd.set_defaults(func=featurize_input)

  # TRAIN FLAGS
  train_cmd = subparsers.add_parser("train",
                  help="Train a model on specified data.")
  group = train_cmd.add_argument_group("load-and-transform")
  group.add_argument("--task-type", default="classification",
                      choices=["classification", "regression"],
                      help="Type of learning task.")
  group.add_argument("--input-transforms", nargs="+", default=[],
                      choices=["normalize", "truncate-outliers"],
                      help="Transforms to apply to input data.")
  group.add_argument("--output-transforms", nargs="+", default=[],
                      choices=["log", "normalize"],
                      help="Transforms to apply to output data.")
  group.add_argument("--feature-types", nargs="+", required=1,
                      help="Types of featurizations to use.")
  group.add_argument("--paths", nargs="+", required=1,
                      help="Paths to input datasets.")
  group.add_argument("--splittype", type=str, default="scaffold",
                       choices=["scaffold", "random", "specified"],
                       help="Type of train/test data-splitting.\n"
                            "scaffold uses Bemis-Murcko scaffolds.\n"
                            "specified requires that split be in original data.")

  group = train_cmd.add_argument_group("model")
  group.add_argument("--mode", default="singletask",
                      choices=["singletask", "multitask"],
                      help="Type of model being built.")
  group.add_argument("--model", required=1,
                      choices=["logistic", "rf_classifier", "rf_regressor",
                      "linear", "ridge", "lasso", "lasso_lars", "elastic_net",
                      "singletask_deep_network", "multitask_deep_network",
                      "3D_cnn"])
  group.add_argument("--n-hidden", type=int, default=500,
                      help="Number of hidden neurons for NN models.")
  group.add_argument("--learning-rate", type=float, default=0.01,
                  help="Learning rate for NN models.")
  group.add_argument("--dropout", type=float, default=0.5,
                  help="Learning rate for NN models.")
  group.add_argument("--n-epochs", type=int, default=50,
                  help="Number of epochs for NN models.")
  group.add_argument("--batch-size", type=int, default=32,
                  help="Number of examples per minibatch for NN models.")
  group.add_argument("--decay", type=float, default=1e-4,
                  help="Learning rate decay for NN models.")
  group.add_argument("--validation-split", type=float, default=0.0,
                  help="Percent of training data to use for validation.")
  group.add_argument("--weight-positives", type=bool, default=False,
                  help="Weight positive examples to have same total weight as negatives.")

  group = train_cmd.add_argument_group("save")
  group.add_argument("--saved-out", type=str, required=1,
                  help="Location to save trained model.")

  eval_cmd = subparsers.add_parser("eval",
                help="Evaluate trained model on specified data.")
  eval_cmd.add_argument("--paths", nargs="+", required=1,
                      help="Paths to input datasets.")
  eval_cmd.add_argument("--splittype", type=str, default="scaffold",
                       choices=["scaffold", "random", "specified"],
                       help="Type of train/test data-splitting.\n"
                            "scaffold uses Bemis-Murcko scaffolds.\n"
                            "specified requires that split be in original data.")
  eval_cmd.add_argument("--compute-aucs", type=bool, default=False,
                      help="Compute AUC for trained models on test set.")
  eval_cmd.add_argument("--compute-r2s", type=bool, default=False,
                      help="Compute R^2 for trained models on test set.")
  eval_cmd.add_argument("--compute-rms", type=bool, default=False,
                      help="Compute RMS for trained models on test set.")
  eval_cmd.add_argument("--csv-out", type=str, default=None,
                  help="Outputted predictions on the test set.")

  return parser.parse_args(input_args)

def featurize_input(args):
  """Featurizes raw input data."""
  if len(args.fields) != len(args.field_types):
    raise ValueError("number of fields does not equal number of field types")
  out_x_pkl, out_y_pkl, out_sdf = generate_directories(args.name, args.out, 
      args.feature_endpoints)
  df, mols = extract_data(args.input_file, args.input_type, args.fields,
      args.field_types, args.prediction_endpoint, args.smiles_endpoint,
      args.threshold, args.delimiter)
  generate_targets(df, mols, args.prediction_endpoint, args.split_endpoint,
      args.smiles_endpoint, out_y_pkl, out_sdf)
  generate_features(df, args.feature_endpoints, args.smiles_endpoint, out_x_pkl)
  generate_fingerprints(args.name, args.out)
  generate_descriptors(args.name, args.out)

def train_model(args):
  """Builds model from featurized data."""
  paths = args.paths
  targets = get_target_names(paths)
  task_types = {target: args.task_type for target in targets}
  input_transforms = args.input_transforms 
  output_transforms = {target: args.output_transforms for target in targets}

  # TODO(rbharath): The datatype (vector vs. tensor) should be automatically
  # detected in dataset_to_numpy
  datatype = "tensor" if args.model == "3D_cnn" else "vector"
  per_task_data = process_datasets(paths,
      input_transforms, output_transforms, feature_types=args.feature_types, 
      prediction_endpoint=args.prediction_endpoint,
      split_endpoint=args.split_endpoint,
      splittype=args.splittype, weight_positives=args.weight_positives,
      datatype=datatype, mode=args.mode)
  # TODO(rbharath): Bundle training params into a training_param dict that's passed
  # down to these functions.
  if args.model == "singletask_deep_network":
    models = fit_singletask_mlp(per_task_data, task_types, n_hidden=args.n_hidden,
      learning_rate=args.learning_rate, dropout=args.dropout,
      nb_epoch=args.n_epochs, decay=args.decay, batch_size=args.batch_size,
      validation_split=args.validation_split)
  elif args.model == "multitask_deep_network":
    models = fit_multitask_mlp(per_task_data, task_types,
      n_hidden=args.n_hidden, learning_rate = args.learning_rate,
      dropout = args.dropout, batch_size=args.batch_size,
      nb_epoch=args.n_epochs, decay=args.decay,
      validation_split=args.validation_split)
  elif args.model == "3D_cnn":
    models = fit_3D_convolution(train_data, test_data, task_types,
        axis_length=args.axis_length, nb_epoch=args.n_epochs,
        batch_size=args.batch_size)
  else:
    models = fit_singletask_models(per_task_data, args.model, task_types)
  # TODO(rbharath): Save trained model.

def eval_trained_model(args):
  results, aucs, r2s, rms = compute_model_performance(per_task_data, models,
    args.compute_aucs, args.compute_r2s, args.compute_rms) 
  if args.csv_out is not None:
    results_to_csv(results, args.csv_out, task_type=args.task_type)

def main():
  args = parse_args()
  args.func(args)



if __name__ == "__main__":
  main()
