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
from deepchem.utils.dataset import FeaturizedSamples

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

  dataset = FeaturizedSamples(paths=paths)
  train_dataset, test_dataset = dataset.train_test_split(splittype)

  train_dir = os.path.join(data_dir, "train")
  train_arrays = train_dataset.to_arrays(train_dir, mode, feature_types)
  print("Transforming train data.")
  train_arrays.transform_data(input_transforms, output_transforms)

  test_dir = os.path.join(data_dir, "test")
  test_arrays = test_dataset.to_arrays(test_dir, mode, feature_types)
  print("Transforming test data.")
  test_arrays.transform_data(input_transforms, output_transforms)

def undo_normalization(y, y_means, y_stds):
  """Undo the applied normalization transform."""
  return y * y_stds + y_means

def undo_transform(y, y_means, y_stds, output_transforms):
  """Undo transforms on y_pred, W_pred."""
  if not isinstance(output_transforms, list):
    output_transforms = [output_transforms]
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
    raise ValueError("Unsupported output transforms %s." % str(output_transforms))

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
