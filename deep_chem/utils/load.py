"""
Utility functions to load datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gzip
import numpy as np
import os
import cPickle as pickle
from deep_chem.utils.preprocess import transform_outputs
from deep_chem.utils.preprocess import transform_inputs
from deep_chem.utils.preprocess import standardize
from deep_chem.utils.preprocess import split_dataset

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"


def process_datasets(paths, feature_types=None, mode="multitask",
                     splittype="random", target_names=None):
  """Extracts datasets and split into train/test.

  Returns a dict with the following key/value pairs

  features -> X
  mol_ids  -> ids
  target -> (y, W)
  sorted_targets -> sorted_targets

  Parameters
  ----------
  paths: list
    List of paths to Google vs datasets.
  splittype: string
    Must be "random" or "scaffold"
  """
  dataset = load_datasets(paths, feature_types=feature_types, target_names=target_names)
  train, test = split_dataset(dataset, splittype)
  train_dict = standardize(train, mode)
  test_dict = standardize(test, mode)
  return train_dict, test_dict

def load_molecules(paths, feature_types):
  """Load dataset fingerprints and return fingerprints.

  Returns a dictionary that maps smiles strings to dicts that contain
  fingerprints, smiles strings, scaffolds, mol_ids.

  Parameters
  ----------
  paths: list
    List of strings.
  """
  molecules = {}
  for dataset_path in paths:
    for feature_type in feature_types:
      pickle_dir = os.path.join(dataset_path, feature_type)
      pickle_files = os.listdir(pickle_dir)
      if len(pickle_files) == 0:
        raise ValueError("No Pickle Files found to load molecules")
      for pickle_file in pickle_files:
        with gzip.open(os.path.join(pickle_dir, pickle_file), "rb") as f:
          contents = pickle.load(f)
          features, scaffolds, mol_ids = (
              contents["features"], contents["scaffolds"], contents["mol_id"])
          splits = contents["split"] if "split" in contents else None
          for mol in range(len(contents["mol_id"])):
            if mol_ids[mol] not in molecules:
              molecules[mol_ids[mol]] = {"fingerprint": features[mol],
                                         "scaffold": scaffolds[mol],
                                         "mol_id": mol_ids[mol],
                                         "feature_types": [feature_type]}
            if splits is not None:
              molecules[mol_ids[mol]]["split"] = splits[mol]
            elif feature_type not in molecules[mol_ids[mol]]["feature_types"]:
              entry = molecules[mol_ids[mol]]
              entry["fingerprint"] = np.append(
                  molecules[mol_ids[mol]]["fingerprint"], features[mol])
              entry["feature_types"].append(feature_type)
  return molecules

def load_assays(paths, target_dir_name, target_names):
  """Load regression dataset labels from assays.

  Returns a dictionary that maps mol_id's to label vectors.

  Parameters
  ----------
  paths: list
    List of paths to base directory.
  target_dir_name: string
    Name of subdirectory containing assay data.
  """
  labels, splits = {}, {}
  # Compute target names
  for dataset_path in paths:
    target_dir = os.path.join(dataset_path, target_dir_name)
    for target_pickle in os.listdir(target_dir):
      if "pkl.gz" not in target_pickle:
        continue
      with gzip.open(os.path.join(target_dir, target_pickle), "rb") as f:
        contents = pickle.load(f)
        for ind, mol_id in enumerate(contents["mol_id"]):
          if "split" in contents:
            splits[mol_id] = contents["split"][ind]
          else:
            splits[mol_id] = None
          if mol_id not in labels:
            labels[mol_id] = {}
            # Ensure that each target has some entry in dict.
            for target_name in target_names:
              # Set all targets to invalid for now.
              labels[mol_id][target_name] = -1
          for target_name in target_names:
            measurement = contents[target_name][ind]
            try:
              if measurement is None or np.isnan(measurement):
                continue
            except TypeError:
              continue
            labels[mol_id][target_name] = measurement
  return labels, splits

def load_datasets(paths, target_dir_name="targets", feature_types=None,
                  target_names=None):
  """Load both labels and fingerprints.

  Returns a dictionary that maps mol_id's to pairs of (fingerprint, labels)
  where labels is itself a dict that maps target-names to labels.

  Parameters
  ----------
  paths: string or list
    Paths to base directory.
  """
  data = {}
  molecules = load_molecules(paths, feature_types)
  labels, splits = load_assays(paths, target_dir_name, target_names)
  for mol_id in molecules:
    if mol_id not in labels:
      continue
    mol = molecules[mol_id]
    data[mol_id] = {"fingerprint": mol["fingerprint"],
                    "scaffold": mol["scaffold"],
                    "labels": labels[mol_id],
                    "split": splits[mol_id]}
  return data

def ensure_balanced(y, W):
  """Helper function that ensures postives and negatives are balanced."""
  n_samples, n_targets = np.shape(y)
  for target_ind in range(n_targets):
    pos_weight, neg_weight = 0, 0
    for sample_ind in range(n_samples):
      if y[sample_ind, target_ind] == 0:
        neg_weight += W[sample_ind, target_ind]
      elif y[sample_ind, target_ind] == 1:
        pos_weight += W[sample_ind, target_ind]
    assert np.isclose(pos_weight, neg_weight)

def transform_data(data, input_transforms, output_transforms):
  """Transform data labels as specified

  Parameters
  ----------
  paths: list
    List of paths to Google vs datasets.
  output_transforms: dict
    dict mapping target names to list of label transforms. Each list element
    must be None, "log", "normalize", or "log-normalize". The transformations
    are performed in the order specified. An empty list corresponds to no
    transformations. Only for regression outputs.
  """
  trans_dict = {}
  X = transform_inputs(data["features"], input_transforms)
  trans_dict["mol_ids"], trans_dict["features"] = data["mol_ids"], X
  trans_dict["sorted_tasks"] = data["sorted_tasks"]
  for task in data["sorted_tasks"]:
    y, W = data[task]
    y = transform_outputs(y, W, output_transforms)
    trans_dict[task] = (y, W)
  assert trans_dict.keys() == data.keys()
  return trans_dict
