"""
Convenience script to train basic models on supported datasets.
"""
import argparse
import numpy as np
from deep_chem.models.deep import fit_singletask_mlp
from deep_chem.models.deep import fit_multitask_mlp
from deep_chem.models.deep3d import fit_3D_convolution
from deep_chem.models.standard import fit_singletask_models
from deep_chem.utils.load import get_target_names
from deep_chem.utils.load import process_datasets
from deep_chem.utils.evaluate import results_to_csv

# TODO(rbharath): Factor this into subcommands. The interface is too
# complicated now to effectively use.
def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--task-type", default="classification",
                      choices=["classification", "regression"],
                      help="Type of learning task.")
  parser.add_argument("--input-transforms", nargs="+", default=[],
                      choices=["normalize", "truncate-outliers"],
                      help="Transforms to apply to input data.")
  parser.add_argument("--output-transforms", nargs="+", default=[],
                      choices=["log", "normalize"],
                      help="Transforms to apply to output data.")
  parser.add_argument("--feature-types", nargs="+", required=1,
                      choices=["fingerprints", "descriptors", "grid"],
                      help="Types of featurizations to use.")
  parser.add_argument("--paths", nargs="+", required=1,
                      help="Paths to input datasets.")
  parser.add_argument("--mode", default="singletask",
                      choices=["singletask", "multitask"],
                      help="Type of model being built.")
  parser.add_argument("--model", required=1,
                      choices=["logistic", "rf_classifier", "rf_regressor",
                      "linear", "ridge", "lasso", "lasso_lars", "elastic_net",
                      "singletask_deep_network", "multitask_deep_network",
                      "3D_cnn"])
  parser.add_argument("--splittype", type=str, default="scaffold",
                       choices=["scaffold", "random", "specified"],
                       help="Type of train/test data-splitting.\n"
                            "scaffold uses Bemis-Murcko scaffolds.\n"
                            "specified requires that split be in original data.")
  parser.add_argument("--csv-out", type=str, default=None,
                  help="Outputted predictions on the test set.")
  #TODO(rbharath): These two arguments (prediction/split-endpoint) should be
  #moved to process_datataset to simplify the invocation here.
  parser.add_argument("--prediction-endpoint", type=str, required=1,
                       help="Name of measured endpoint to predict.")
  parser.add_argument("--split-endpoint", type=str, default=None,
                       help="Name of endpoint specifying train/test split.")
  parser.add_argument("--n-hidden", type=int, default=500,
                      help="Number of hidden neurons for NN models.")
  parser.add_argument("--learning-rate", type=float, default=0.01,
                  help="Learning rate for NN models.")
  parser.add_argument("--dropout", type=float, default=0.5,
                  help="Learning rate for NN models.")
  parser.add_argument("--n-epochs", type=int, default=50,
                  help="Number of epochs for NN models.")
  parser.add_argument("--batch-size", type=int, default=32,
                  help="Number of examples per minibatch for NN models.")
  parser.add_argument("--decay", type=float, default=1e-4,
                  help="Learning rate decay for NN models.")
  parser.add_argument("--validation-split", type=float, default=0.0,
                  help="Percent of training data to use for validation.")
  parser.add_argument("--weight-positives", type=bool, default=False,
                  help="Weight positive examples to have same total weight as negatives.")
  # TODO(rbharath): Remove this once debugging is complete.
  parser.add_argument("--num-to-train", type=int, default=None,
                  help="Number of datasets to train on. Only for debug.")
  parser.add_argument("--axis-length", type=int, default=32,
                  help="Size of a grid axis for 3D CNN input.")
  return parser.parse_args(input_args)

def main():
  args = parse_args()
  paths = {}

  paths = args.paths

  targets = get_target_names(paths)
  task_types = {target: args.task_type for target in targets}
  input_transforms = args.input_transforms 
  output_transforms = {target: args.output_transforms for target in targets}

  datatype = "tensor" if args.model == "3D_cnn" else "vector"
  processed = process_datasets(paths,
      input_transforms, output_transforms, feature_types=args.feature_types, 
      prediction_endpoint=args.prediction_endpoint,
      split_endpoint=args.split_endpoint,
      splittype=args.splittype, weight_positives=args.weight_positives,
      datatype=datatype, mode=args.mode)
  if args.mode == "multitask":
    train_data, test_data = processed
  else:
    per_task_data = processed
  # TODO(rbharath): Bundle training params into a training_param dict that's passed
  # down to these functions.
  if args.model == "singletask_deep_network":
    results = fit_singletask_mlp(per_task_data, task_types, n_hidden=args.n_hidden,
      learning_rate=args.learning_rate, dropout=args.dropout,
      nb_epoch=args.n_epochs, decay=args.decay, batch_size=args.batch_size,
      validation_split=args.validation_split,
      num_to_train=args.num_to_train)
  elif args.model == "multitask_deep_network":
    results = fit_multitask_mlp(train_data, test_data, task_types,
      n_hidden=args.n_hidden, learning_rate = args.learning_rate,
      dropout = args.dropout, batch_size=args.batch_size,
      nb_epoch=args.n_epochs, decay=args.decay,
      validation_split=args.validation_split)
  elif args.model == "3D_cnn":
    results = fit_3D_convolution(train_data, test_data, task_types,
        axis_length=args.axis_length, nb_epoch=args.n_epochs,
        batch_size=args.batch_size)
  else:
    results = fit_singletask_models(per_task_data, args.model, task_types,
                                    num_to_train=args.num_to_train)
  
  if args.csv_out is not None:
    results_to_csv(results, args.csv_out, task_type=args.task_type)

if __name__ == "__main__":
  main()
