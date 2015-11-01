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

# TODO(rbharath): Factor this into subcommands. The interface is too
# complicated now to effectively use.
def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  #parser.add_argument("--datasets", nargs="+", required=1,
  #                    choices=["muv", "pcba", "dude", "pfizer", "globavir", "pdbbind"],
  #                    help="Name of dataset to process.")
  #parser.add_argument("--dataset-names", nargs="+", required=1,
  #                    help="Names of datasets to process.")
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
                      help = "Paths to input datasets.")
  parser.add_argument("--model", required=1,
                      choices=["logistic", "rf_classifier", "rf_regressor",
                      "linear", "ridge", "lasso", "lasso_lars", "elastic_net",
                      "singletask_deep_network", "multitask_deep_network",
                      "3D_cnn"])
  parser.add_argument("--splittype", type=str, default="scaffold",
                       choices=["scaffold", "random"],
                       help="Type of cross-validation data-splitting.")
  parser.add_argument("--prediction-endpoint", type=str, required=1,
                       help="Name of measured endpoint to predict.")
  # TODO(rbharath): There should be a way to directly compute n-input from the
  # provided feature-types?
  parser.add_argument("--n-inputs", type=int, default=1024,
                      help="Number of input features for models.")
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

  print paths
  targets = get_target_names(paths)
  task_types = {target: args.task_type for target in targets}
  input_transforms = args.input_transforms 
  print "input_transforms"
  print input_transforms
  output_transforms = {target: args.output_transforms for target in targets}

  # TODO(rbharath): Too many settings are explicitly passed down here. Is there
  # a good way to pass down configuration parameters into the invocations
  # below.
  if args.model == "singletask_deep_network":
    fit_singletask_mlp(paths, task_types, input_transforms, output_transforms,
      prediction_endpoint=args.prediction_endpoint,
      feature_types=args.feature_types,
      splittype=args.splittype, 
      n_inputs=args.n_inputs,
      n_hidden=args.n_hidden,
      learning_rate=args.learning_rate, dropout=args.dropout,
      nb_epoch=args.n_epochs, decay=args.decay, batch_size=args.batch_size,
      validation_split=args.validation_split,
      weight_positives=args.weight_positives, num_to_train=args.num_to_train)
  elif args.model == "multitask_deep_network":
    fit_multitask_mlp(paths, task_types, input_transforms, output_transforms,
      prediction_endpoint=args.prediction_endpoint,
      feature_types=args.feature_types,
      splittype=args.splittype,
      n_inputs=args.n_inputs,
      n_hidden=args.n_hidden, learning_rate =
      args.learning_rate, dropout = args.dropout, batch_size=args.batch_size,
      nb_epoch=args.n_epochs, decay=args.decay,
      validation_split=args.validation_split,
      weight_positives=args.weight_positives)
  elif args.model == "3D_cnn":
    fit_3D_convolution(paths, task_types, input_transforms, output_transforms,
        feature_types=args.feature_types,
        prediction_endpoint=args.prediction_endpoint,
        axis_length=args.axis_length, nb_epoch=args.n_epochs,
        batch_size=args.batch_size)
  else:
    print args.prediction_endpoint
    fit_singletask_models(paths, args.model, task_types,
        input_transforms, output_transforms, prediction_endpoint=args.prediction_endpoint,
        feature_types=args.feature_types, splittype=args.splittype,
        num_to_train=args.num_to_train)

if __name__ == "__main__":
  main()
