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
from deep_chem.utils.preprocess import tensor_dataset_to_numpy
from deep_chem.utils.preprocess import multitask_to_singletask
from deep_chem.utils.preprocess import split_dataset
from deep_chem.utils.preprocess import to_arrays
from vs_utils.utils import ScaffoldGenerator

def process_datasets(paths, input_transforms, output_transforms,
    prediction_endpoint=None, split_endpoint=None, datatype="vector",
    feature_types=["fingerprints"], mode="multitask", splittype="random",
    seed=None, weight_positives=True):
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
  dataset = load_and_transform_dataset(paths, input_transforms, output_transforms,
      prediction_endpoint, split_endpoint=split_endpoint,
      feature_types=feature_types, weight_positives=weight_positives)
  arrays = {}
  if mode == "singletask":
    singletask = multitask_to_singletask(dataset)
    for target in singletask:
      data = singletask[target]
      if len(data) == 0:
        continue
      train, test = split_dataset(dataset, splittype)
      train_data, test_data = to_arrays(train, test, datatype)
      arrays[target] = (train_data, test_data)
  elif mode == "multitask":
    train, test = split_dataset(dataset, splittype)
    train_data, test_data = to_arrays(train, test, datatype)
    arrays["all"] = (train_data, test_data)
  else:
    raise ValueError("Unsupported mode for process_datasets.")
  return arrays


def load_molecules(paths, feature_types=["fingerprints"]):
  """Load dataset fingerprints and return fingerprints.

  Returns a dictionary that maps smiles strings to dicts that contain
  fingerprints, smiles strings, scaffolds, mol_ids.

  TODO(rbharath): This function assumes that all datapoints are uniquely keyed
  by smiles strings. This doesn't hold true for the pdbbind dataset. Need to find
  a more general indexing mechanism.

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
          for mol in range(len(contents["smiles"])):
            if smiles[mol] not in molecules:
              molecules[smiles[mol]] = {"fingerprint": features[mol],
                                        "scaffold": scaffolds[mol],
                                        "mol_id": mol_ids[mol],
                                        "feature_types": [feature_type]}
            if splits is not None:
              molecules[smiles[mol]]["split"] = splits[mol]
            # TODO(rbharath): Our processing pipeline sometimes makes different
            # molecules look the same (due to bugs in how we hydrogenate for
            # example). Fix these bugs in our processing pipeline.
            elif feature_type not in molecules[smiles[mol]]["feature_types"]:
              entry = molecules[smiles[mol]]
              entry["fingerprint"] = np.append(
                  molecules[smiles[mol]]["fingerprint"], features[mol])
              entry["feature_types"].append(feature_type)
  return molecules 

def get_target_names(paths, target_dir_name="targets"):
  """Get names of targets in provided collections.

  Parameters
  ----------
  paths: list 
    List of paths to base directory.
  """
  target_names = []
  for dataset_path in paths:
    target_dir = os.path.join(dataset_path, target_dir_name)
    target_names += [target_pickle.split(".")[0]
        for target_pickle in os.listdir(target_dir)
        if "pkl.gz" in target_pickle]
  return target_names

def load_assays(paths, prediction_endpoint, split_endpoint=None, target_dir_name="targets"):
  """Load regression dataset labels from assays.

  Returns a dictionary that maps smiles strings to label vectors.

  TODO(rbharath): Remove the use of smiles as unique identifier

  Parameters
  ----------
  paths: list 
    List of paths to base directory.
  target_dir_name: string
    Name of subdirectory containing assay data.
  """
  labels, splits = {}, {}
  # Compute target names
  target_names = get_target_names(paths, target_dir_name)
  for dataset_path in paths:
    target_dir = os.path.join(dataset_path, target_dir_name)
    for target_pickle in os.listdir(target_dir):
      if "pkl.gz" not in target_pickle:
        continue
      target_name = target_pickle.split(".")[0]
      with gzip.open(os.path.join(target_dir, target_pickle), "rb") as f:
        contents = pickle.load(f)
        if prediction_endpoint not in contents:
          raise ValueError("Prediction Endpoint Missing.")
        for ind, smiles in enumerate(contents["smiles"]):
          measurement = contents[prediction_endpoint][ind]
          if split_endpoint is not None:
            splits[smiles] = contents[split_endpoint][ind]
          else:
            splits[smiles] = None
          # TODO(rbharath): There is some amount of duplicate collisions
          # due to choice of smiles generation. Look into this more
          # carefully and see if the underlying issues are fundamental..
          try:
            if measurement is None or np.isnan(measurement):
              continue
          except TypeError:
            continue
          if smiles not in labels:
            labels[smiles] = {}
            # Ensure that each target has some entry in dict.
            for name in target_names:
              # Set all targets to invalid for now.
              labels[smiles][name] = -1
          labels[smiles][target_name] = measurement 
  return labels, splits

def load_datasets(paths, prediction_endpoint, split_endpoint, target_dir_name="targets",
    feature_types=["fingerprints"]):
  """Load both labels and fingerprints.

  Returns a dictionary that maps smiles to pairs of (fingerprint, labels)
  where labels is itself a dict that maps target-names to labels.

  Parameters
  ----------
  paths: string or list
    Paths to base directory.
  """
  data = {}
  molecules = load_molecules(paths, feature_types)
  labels, splits = load_assays(paths, prediction_endpoint, split_endpoint, target_dir_name)
  for ind, smiles in enumerate(molecules):
    if smiles not in labels:
      continue
    mol = molecules[smiles]
    data[smiles] = {"fingerprint": mol["fingerprint"],
                    "scaffold": mol["scaffold"],
                    "labels": labels[smiles],
                    "split": splits[smiles]}
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

def load_and_transform_dataset(paths, input_transforms, output_transforms,
    prediction_endpoint, split_endpoint=None, labels_endpoint="labels", weight_positives=True,
    datatype="tensor", feature_types=["fingerprints"]):
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
  dataset = load_datasets(paths, prediction_endpoint, split_endpoint,
      feature_types=feature_types)
  if datatype == "vector":
    X, y, W = dataset_to_numpy(dataset, weight_positives=weight_positives)
  elif datatype == "tensor":
    X, y, W = tensor_dataset_to_numpy(dataset)
  y = transform_outputs(y, W, output_transforms,
      weight_positives=weight_positives)
  X = transform_inputs(X, input_transforms)
  trans_data = {}
  sorted_smiles = sorted(dataset.keys())
  sorted_targets = sorted(output_transforms.keys())
  for s_index, smiles in enumerate(sorted_smiles):
    datapoint = dataset[smiles]
    labels = {}
    for t_index, target in enumerate(sorted_targets):
      if W[s_index][t_index] == 0:
        labels[target] = -1
      else:
        labels[target] = y[s_index][t_index]
    datapoint[labels_endpoint] = labels
    datapoint["fingerprint"] = X[s_index]

    trans_data[smiles] = datapoint 
  return trans_data
