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
from deep_chem.utils.preprocess import get_default_task_types_and_transforms
from deep_chem.utils.preprocess import get_default_descriptor_transforms

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--assay', required=1,
                      help='Assay ID.')
  parser.add_argument('--dataset', required=1, choices=['muv', 'pcba', 'dude', 'pfizer'],
                      help='Name of dataset to process.')
  parser.add_argument('--model', required=1, nargs="+",
                      choices=["logistic", "rf_classifier", "single_task_deep_network"])
  return parser.parse_args(input_args)

def main():
  args = parse_args()
  if args.dataset == "muv":
    path = "/home/rbharath/vs-datasets/muv"
  elif args.dataset == "pcba":
    path = "/home/rbharath/vs-datasets/pcba"
  elif args.dataset == "dude":
    path = "/home/rbharath/vs-datasets/dude"
  # TODO(rbharath): The pfizer dataset is private. Remove this before the
  # public release of the code.
  elif args.dataset == "pfizer":
    path = "/home/rbharath/private-datasets/pfizer"

  task_types, task_transforms = get_default_task_types_and_transforms(
    {args.dataset: path})
  desc_transforms = get_default_descriptor_transforms()

  if len(args.model) == 1:
    model = args.model[0]
    fit_singletask_models([path], model, task_types,
        task_transforms, splittype="scaffold")

  #fit_multitask_mlp([muv_path, pfizer_path], task_types, task_transforms,
  #  desc_transforms, splittype="scaffold", add_descriptors=False,
  #  desc_weight=0.1, n_hidden=500, learning_rate = .01, dropout = .5,
  #  nb_epoch=50, decay=1e-4, validation_split=0.01)

  #fit_multitask_mlp([muv_path, pfizer_path], task_types, task_transforms,
  #  desc_transforms, splittype="scaffold", add_descriptors=False, n_hidden=500,
  #  nb_epoch=40, learning_rate=0.01, decay=1e-4, dropout = .5)
  #fit_multitask_mlp([dude_path], task_types, task_transforms,
  #  desc_transforms, splittype="scaffold", add_descriptors=False, n_hidden=500,
  #  nb_epoch=40, learning_rate=0.01, decay=1e-4, dropout = .5)
  #fit_multitask_mlp([muv_path], task_types, task_transforms, desc_transforms,
  #  splittype="scaffold", add_descriptors=False, n_hidden=500,
  #  learning_rate=.01, dropout=.5, nb_epoch=30, decay=1e-4)
  #fit_singletask_mlp([muv_path], task_types, task_transforms, desc_transforms,
  #  splittype="scaffold", add_descriptors=False, n_hidden=500,
  #  learning_rate=.01, dropout=.5, nb_epoch=30, decay=1e-4)


if __name__ == "__main__":
  main()
