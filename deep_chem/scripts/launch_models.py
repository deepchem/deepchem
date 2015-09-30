"""
Convenience script to train basic models on supported datasets.
"""
import argparse
import numpy as np
from deep_chem.models.deep import fit_singletask_mlp
from deep_chem.models.deep import fit_multitask_mlp
from deep_chem.models.deep import train_multitask_model
from deep_chem.models.standard import fit_singletask_models
from deep_chem.models.standard import fit_multitask_rf
from deep_chem.utils.analysis import compare_datasets
from deep_chem.utils.evaluate import eval_model
from deep_chem.utils.evaluate import compute_roc_auc_scores
from deep_chem.utils.evaluate import compute_r2_scores
from deep_chem.utils.evaluate import compute_rms_scores
from deep_chem.utils.load import get_target_names
from deep_chem.utils.load import load_datasets
from deep_chem.utils.load import load_and_transform_dataset
from deep_chem.utils.preprocess import dataset_to_numpy
from deep_chem.utils.preprocess import train_test_random_split
from deep_chem.utils.preprocess import train_test_scaffold_split
from deep_chem.utils.preprocess import scaffold_separate
from deep_chem.utils.preprocess import multitask_to_singletask
from deep_chem.utils.load import get_default_task_types_and_transforms
from deep_chem.utils.preprocess import get_default_descriptor_transforms

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--datasets', required=1, nargs="+",
                      choices=['muv', 'pcba', 'dude', 'pfizer'],
                      help='Name of dataset to process.')
  parser.add_argument("--paths", required=1, nargs="+",
                      help = "Paths to input datasets.")
  parser.add_argument('--model', required=1,
                      choices=["logistic", "rf_classifier", "rf_regressor",
                      "linear", "ridge", "lasso", "lasso_lars", "elastic_net",
                      "singletask_deep_network", "multitask_deep_network"])
  parser.add_argument("--splittype", type=str, default="scaffold",
                       choices=["scaffold", "random"],
                       help="Type of cross-validation data-splitting.")
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
  return parser.parse_args(input_args)

def main():
  args = parse_args()
  paths = {}
  for dataset, path in zip(args.datasets, args.paths):
    paths[dataset] = path

  task_types, task_transforms = get_default_task_types_and_transforms(paths)
  desc_transforms = get_default_descriptor_transforms()

  if args.model == "singletask_deep_network":
    fit_singletask_mlp(paths.values(), task_types, task_transforms,
      desc_transforms, splittype=args.splittype, add_descriptors=False,
      n_hidden=args.n_hidden, learning_rate=args.learning_rate,
      dropout=args.dropout, nb_epoch=args.n_epochs, decay=args.decay,
      batch_size=args.batch_size,
      validation_split=args.validation_split)
  elif args.model == "multitask_deep_network":
    fit_multitask_mlp(paths.values(), task_types, task_transforms,
      desc_transforms, splittype=args.splittype, add_descriptors=False,
      n_hidden=args.n_hidden, learning_rate = args.learning_rate, dropout = args.dropout,
      batch_size=args.batch_size,
      nb_epoch=args.n_epochs, decay=args.decay, validation_split=args.validation_split)
  else:
    fit_singletask_models(paths.values(), args.model, task_types,
        task_transforms, splittype="scaffold")

if __name__ == "__main__":
  main()
