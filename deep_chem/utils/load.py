"""
Utility functions to load datasets.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import gzip
import numpy as np
import os
import cPickle as pickle
from deep_chem.utils.preprocess import transform_outputs
from deep_chem.utils.preprocess import transform_inputs
from deep_chem.utils.preprocess import dataset_to_numpy
from deep_chem.utils.preprocess import multitask_to_singletask
from deep_chem.utils.preprocess import split_dataset
from deep_chem.utils.preprocess import to_arrays
from vs_utils.utils import ScaffoldGenerator

def process_datasets(paths, input_transforms, output_transforms,
    feature_types=["fingerprints"], mode="multitask",
    splittype="random", seed=None, weight_positives=True, target_names=[]):
  """Extracts datasets and split into train/test.

  Returns a dict that maps target names to tuples.

  Parameters
  ----------
  paths: list 
    List of paths to Google vs datasets. 
  output_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  splittype: string
    Must be "random" or "scaffold"
  seed: int
    Seed used for random splits.
  """
  dataset = load_datasets(paths, feature_types=feature_types, target_names=target_names)
  train_dict, test_dict = {}, {}
  if mode == "singletask":
    singletask = multitask_to_singletask(dataset)
    for target in singletask:
      data = singletask[target]
      if len(data) == 0:
        continue
      train, test = split_dataset(dataset, splittype)
      train_dict[target], test_dict[target] = to_arrays(train, test)
  elif mode == "multitask":
    train, test = split_dataset(dataset, splittype)
    train_data, test_data = to_arrays(train, test)
    train_dict["all"], test_dict["all"] = train_data, test_data
  else:
    raise ValueError("Unsupported mode for process_datasets.")
  print "Shape of Xtrain"
  target = train_dict.itervalues().next()
  print np.shape(target[1])
  return train_dict, test_dict 

def load_molecules(paths, feature_types=["fingerprints"]):
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
          smiles, features, scaffolds, mol_ids = (
              contents["smiles"], contents["features"],
              contents["scaffolds"], contents["mol_id"])
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

#def get_target_names(paths, target_dir_name="targets"):
#  """Get names of targets in provided collections.
#
#  Parameters
#  ----------
#  paths: list 
#    List of paths to base directory.
#  """
#  target_names = []
#  for dataset_path in paths:
#    target_dir = os.path.join(dataset_path, target_dir_name)
#    target_names += [target_pickle.split(".")[0]
#        for target_pickle in os.listdir(target_dir)
#        if "pkl.gz" in target_pickle]
#  return target_names

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

def load_datasets(paths, target_dir_name="targets", feature_types=["fingerprints"],
                  target_names=[]):
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
  for ind, id in enumerate(molecules):
    if id not in labels:
      continue
    mol = molecules[id]
    data[id] = {"fingerprint": mol["fingerprint"],
                "scaffold": mol["scaffold"],
                "labels": labels[id],
                "split": splits[id]}
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
  for target in data:
    ids, X, y, W = data[target]
    y = transform_outputs(y, W, output_transforms)
    X = transform_inputs(X, input_transforms)
    trans_dict[target] = (ids, X, y, W)
  return trans_dict
