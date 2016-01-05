"""
Utility functions to preprocess datasets before building models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import numpy as np
import warnings
from glob import glob
import pandas as pd
import os
import multiprocessing as mp

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def get_task_type(model_name):
  """
  Given model type, determine if classifier or regressor.
  """
  if model_name in ["logistic", "rf_classifier", "singletask_deep_classifier",
                    "multitask_deep_classifier"]:
    return "classification"
  else:
    return "regression"

def train_test_split(paths, input_transforms, output_transforms,
                     feature_types, splittype, mode, data_dir):
  """Saves transformed model."""

  #TODO(enf/rbharath): Scaffold split is completely broken here.
  dataset = FeaturizedDataset(paths=paths)
  train_dataset, test_dataset = dataset.train_test_split(splittype)

  train_dir = os.path.join(data_dir, "train")
  train_arrays = train_dataset.to_arrays(train_dir, mode, feature_types)
  print("Transforming train data.")
  train_arrays.transform_data(input_transforms, output_transforms)

  test_dir = os.path.join(data_dir, "test")
  test_dataset.write(test_dir, mode, feature_types)
  print("Transforming test data.")
  test_arrays.transform_data(input_transforms, output_transforms)

'''
  print("About to train/test split dataset")
  train_files, test_files = get_train_test_files(paths, splittype)
  train_metadata = write_dataset(train_files, data_dir, mode, feature_types)
  train_metadata["split"] = "train"
  test_metadata = write_dataset(test_files, data_dir, mode, feature_types)
  test_metadata["split"] = "test"

  metadata = pd.concat([train_metadata, test_metadata])
  metadata["input_transforms"] = ",".join(input_transforms)
  metadata["output_transforms"] = ",".join(output_transforms)

  metadata = transform_data(metadata, input_transforms, output_transforms)

  metadata_filename = get_metadata_filename(data_dir)
  print("Saving metadata file to %s" % metadata_filename)
  save_sharded_dataset(metadata, metadata_filename)
  print("Saved metadata.")
'''


def undo_normalization(y, y_means, y_stds):
  """Undo the applied normalization transform."""
  y = y * y_means + y_stds
  return y * y_means + y_stds

def undo_transform(y, y_means, y_stds, output_transforms):
  """Undo transforms on y_pred, W_pred."""
  output_transforms = [output_transforms]
  print(output_transforms)
  if (output_transforms == [""] or output_transforms == ['']
    or output_transforms == []):
    return y
  elif output_transforms == ["log"]:
    return np.exp(y)
  elif output_transforms == ["normalize"]:
    return undo_normalization(y, y_means, y_stds)
  elif output_transforms == ["log", "normalize"]:
    return np.exp(undo_normalization(y, y_means, y_stds))
  else:
    raise ValueError("Unsupported output transforms.")

#todo(enf/rbharath): this is completely broken.

'''
def multitask_to_singletask(dataset):
  """transforms a multitask dataset to a singletask dataset.

  returns a dictionary which maps target names to datasets, where each
  dataset is itself a dict that maps identifiers to
  (fingerprint, scaffold, dict) tuples.

  parameters
  ----------
  dataset: dict
    dictionary of type produced by load_datasets
  """
  # generate single-task data structures
  labels = dataset.itervalues().next()["labels"]
  sorted_targets = sorted(labels.keys())
  singletask_features = []
  singletask_labels = {target: [] for target in sorted_targets}
  # populate the singletask datastructures
  sorted_ids = sorted(dataset.keys())
  for mol_id in sorted_ids:
    datapoint = dataset[mol_id]
    labels = datapoint["labels"]
    singletask_features.append(datapoint["fingeprint"])
    for target in sorted_targets:
      if labels[target] == -1:
        continue
      else:
        singletask_labels[target].append(labels[target])
  return singletask_features, singletask_labels
'''

#todo(enf/rbharath): completly broken as well.
'''
def split_dataset(dataset, splittype, seed=none):
  """split provided data using specified method."""
  if splittype == "random":
    train, test = train_test_random_split(dataset, seed=seed)
  elif splittype == "scaffold":
    train, test = train_test_scaffold_split(dataset)
  elif splittype == "specified":
    train, test = train_test_specified_split(dataset)
  else:
    raise valueerror("improper splittype.")
  return train, test

def train_test_specified_split(dataset):
  """split provided data due to splits in origin data."""
  train, test = {}, {}
  for mol_id, datapoint in dataset.iteritems():
    if "split" not in datapoint:
      raise valueerror("missing required split information.")
    if datapoint["split"].lower() == "train":
      train[mol_id] = datapoint
    elif datapoint["split"].lower() == "test":
      test[mol_id] = datapoint
  return train, test

def train_test_random_split(dataset, frac_train=.8, seed=none):
  """splits provided data into train/test splits randomly.

  performs a random 80/20 split of the data into train/test. returns two
  dictionaries

  parameters
  ----------
  dataset: dict
    a dictionary of type produced by load_datasets.
  frac_train: float
    proportion of data in train set.
  seed: int (optional)
    seed to initialize np.random.
  """
  np.random.seed(seed)
  shuffled = np.random.permutation(dataset.keys())
  train_cutoff = np.floor(frac_train * len(shuffled))
  train_keys, test_keys = shuffled[:train_cutoff], shuffled[train_cutoff:]
  train, test = {}, {}
  for key in train_keys:
    train[key] = dataset[key]
  for key in test_keys:
    test[key] = dataset[key]
  return train, test

def train_test_scaffold_split(dataset, frac_train=.8):
  """splits provided data into train/test splits by scaffold.

  groups the largest scaffolds into the train set until the size of the
  train set equals frac_train * len(dataset). adds remaining scaffolds
  to test set. the idea is that the test set contains outlier scaffolds,
  and thus serves as a hard test of generalization capability for the
  model.

  parameters
  ----------
  dataset: dict
    a dictionary of type produced by load_datasets.
  frac_train: float
    the fraction (between 0 and 1) of the data to use for train set.
  """
  scaffolds = scaffold_separate(dataset)
  train_size = frac_train * len(dataset)
  train, test = {}, {}
  for elements in scaffolds:
    # if adding this scaffold makes the train_set too big, add to test set.
    if len(train) + len(elements) > train_size:
      for elt in elements:
        test[elt] = dataset[elt]
    else:
      for elt in elements:
        train[elt] = dataset[elt]
  return train, test

def scaffold_separate(dataset):
  """splits provided data by compound scaffolds.

  returns a list of pairs (scaffold, [identifiers]), where each pair
  contains a scaffold and a list of all identifiers for compounds that
  share that scaffold. the list will be sorted in decreasing order of
  number of compounds.

  parameters
  ----------
  dataset: dict
    a dictionary of type produced by load_datasets.
  """
  scaffolds = {}
  for mol_id in dataset:
    datapoint = dataset[mol_id]
    scaffold = datapoint["scaffold"]
    if scaffold not in scaffolds:
      scaffolds[scaffold] = [mol_id]
    else:
      scaffolds[scaffold].append(mol_id)
  # sort from largest to smallest scaffold sets
  return [elt for (scaffold, elt) in sorted(scaffolds.items(), key=lambda x: -len(x[1]))]
'''
