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
from deep_chem.utils.save import load_sharded_dataset
from deep_chem.utils.save import save_sharded_dataset
from functools import partial

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

def get_train_test_files(paths, train_proportion=0.8):
  """
  Randomly split files into train and test.
  """
  all_files = []
  for path in paths:
    all_files += glob(os.path.join(path, "*.joblib"))
  train_indices = list(np.random.choice(len(all_files), int(len(all_files)*train_proportion)))
  test_indices = list(set(range(len(all_files)))-set(train_indices))

  train_files = [all_files[i] for i in train_indices]
  test_files = [all_files[i] for i in test_indices]
  return train_files, test_files

def get_metadata_filename(data_dir):
  """
  Get standard location for metadata file.
  """
  metadata_filename = os.path.join(data_dir, "metadata.joblib")
  return metadata_filename

def train_test_split(paths, output_transforms, input_transforms,
                     feature_types, splittype, mode, data_dir):
  """Saves transformed model."""

  #TODO(enf/rbharath): Scaffold split is completely broken here.

  #TODO(enf/rbharath): Transforms are also completely broken here.

  #TODO(enf/rbharath): Ability to concat vectorial features is broken.

  print("About to train/test split dataset")
  train_files, test_files = get_train_test_files(paths)
  print("train_files")
  print(train_files)
  print("test_files")
  print(test_files)
  print("About to write numpy arrays for train & test")
  train_metadata = write_dataset(train_files, data_dir, mode)
  train_metadata["split"] = "train"
  test_metadata = write_dataset(test_files, data_dir, mode)
  test_metadata["split"] = "test"

  metadata = pd.concat([train_metadata, test_metadata])
  print("metadata[:3]")
  print(metadata[:3])
  metadata['input_transforms'] = ",".join(input_transforms)
  metadata['output_transforms'] = ",".join(output_transforms)

  metadata_filename = get_metadata_filename(data_dir)
  print("Saving metadata file to %s" % metadata_filename)
  save_sharded_dataset(metadata, metadata_filename)
  print("Saved metadata.")

  '''
  print("Starting transform_data")
  trans_train_dict = transform_data(
      train_dict, input_transforms, output_transforms)
  print("Finished transform_data on train")
  trans_test_dict = transform_data(test_dict, input_transforms, output_transforms)
  print("Finished transform_data on test")
  transforms = {"input_transforms": input_transforms,
                "output_transform": output_transforms}
  stored_train = {"raw": train_dict,
                  "transformed": trans_train_dict,
                  "transforms": transforms}
  stored_test = {"raw": test_dict,
                 "transformed": trans_test_dict,
                 "transforms": transforms}
  print("About to save dataset..")
  save_sharded_dataset(stored_train, train_out)
  save_sharded_dataset(stored_test, test_out)
  '''

def write_dataset_single(df_file, out_dir, mode):
  df = load_sharded_dataset(df_file)
  task_names = get_sorted_task_names(df)
  ids, X, y, w = df_to_numpy(df, mode)
  basename = os.path.splitext(os.path.basename(df_file))[0]
  out_X = os.path.join(out_dir, "%s-X.joblib" % basename)
  out_y = os.path.join(out_dir, "%s-y.joblib" % basename)
  out_w = os.path.join(out_dir, "%s-w.joblib" % basename)
  out_ids = os.path.join(out_dir, "%s-ids.joblib" % basename)

  save_sharded_dataset(X, out_X)
  save_sharded_dataset(y, out_y)
  save_sharded_dataset(w, out_w)
  save_sharded_dataset(ids, out_ids)
  return([df_file, task_names, out_ids, out_X, out_y, out_w])

def write_dataset(df_files, out_dir, mode):
  """
  Turns featurized dataframes into numpy files, writes them & metadata to disk.
  """
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  write_dataset_single_partial = partial(write_dataset_single, out_dir=out_dir, mode=mode)

  pool = mp.Pool(mp.cpu_count())
  metadata_rows = pool.map(write_dataset_single_partial, df_files)
  pool.terminate()

  metadata_df = pd.DataFrame(metadata_rows, 
                             columns=('df_file', 'task_names', 'ids', 'X', 'y', 'w'))
  print("metadata_df[:3]")
  print(metadata_df[:3])
  return metadata_df

def get_sorted_task_names(df):
  """
  Given metadata df, return sorted names of tasks.
  """
  column_names = df.keys()
  task_names = (set(column_names) - 
                set(["mol_id", "smiles", "split", "features", "descriptors", "fingerprints"]))
  return sorted(list(task_names))

def df_to_numpy(df, mode):
  """Transforms a set of tensor data into standard set of numpy arrays"""
  # Perform common train/test split across all tasks
  n_samples = df.shape[0]
  sorted_tasks = get_sorted_task_names(df)
  n_tasks = len(sorted_tasks)

  y = df[sorted_tasks].values
  w = np.ones((n_samples, n_tasks))
  w[np.where(y=='')] = 0

  tensors = []
  for i, datapoint in df.iterrows():
    features = datapoint["features"]
    tensors.append(np.squeeze(features))

  X = np.stack(tensors)
  sorted_ids = df['mol_id']
  return sorted_ids, X, y, w

def transform_inputs(X, input_transforms):
  """Transform the input feature data."""
  # Copy X up front to have non-destructive updates.
  if not input_transforms:
    return X

  X = np.copy(X)
  if len(np.shape(X)) == 2:
    n_features = np.shape(X)[1]
  else:
    raise ValueError("Only know how to transform vectorial data.")
  Z = np.zeros(np.shape(X))
  # Meant to be done after normalize
  trunc = 5
  for feature in range(n_features):
    feature_data = X[:, feature]
    for input_transform in input_transforms:
      if input_transform == "normalize-and-truncate":
        if np.amax(feature_data) > trunc or np.amin(feature_data) < -trunc:
          mean, std = np.mean(feature_data), np.std(feature_data)
          feature_data = feature_data - mean
          if std != 0.:
            feature_data = feature_data / std
          feature_data[feature_data > trunc] = trunc
          feature_data[feature_data < -trunc] = -trunc
          if np.amax(feature_data) > trunc or np.amin(feature_data) < -trunc:
            raise ValueError("Truncation failed on feature %d" % feature)
      else:
        raise ValueError("Unsupported Input Transform")
    Z[:, feature] = feature_data
  return Z

def undo_normalization(y_orig, y_pred):
  """Undo the applied normalization transform."""
  old_mean = np.mean(y_orig)
  old_std = np.std(y_orig)
  return y_pred * old_std + old_mean

def undo_transform_outputs(y_raw, y_pred, output_transforms):
  """Undo transforms on y_pred, W_pred."""
  if output_transforms == []:
    return y_pred
  elif output_transforms == ["log"]:
    return np.exp(y_pred)
  elif output_transforms == ["normalize"]:
    return undo_normalization(y_raw, y_pred)
  elif output_transforms == ["log", "normalize"]:
    return np.exp(undo_normalization(np.log(y_raw), y_pred))
  else:
    raise ValueError("Unsupported output transforms.")

def transform_outputs(y, W, output_transforms):
  """Tranform the provided outputs

  Parameters
  ----------
  y: ndarray
    Labels
  W: ndarray
    Weights
  output_transforms: list
    List of specified transforms (must be "log", "normalize"). The
    transformations are performed in the order specified. An empty list
    corresponds to no transformations. Only for regression outputs.
  """
  # Copy y up front so we have non-destructive updates
  y = np.copy(y)
  #if len(np.shape(y)) == 1:
  #  n_tasks = 1
  #elif len(np.shape(y)) == 2:
  if len(np.shape(y)) == 2:
    n_tasks = np.shape(y)[1]
  else:
    raise ValueError("y must be of shape (n_samples,n_targets)")
  for task in range(n_tasks):
    for output_transform in output_transforms:
      if output_transform == "log":
        y[:, task] = np.log(y[:, task])
      elif output_transform == "normalize":
        task_data = y[:, task]
        if task < n_tasks:
          # Only elements of y with nonzero weight in W are true labels.
          nonzero = (W[:, task] != 0)
        else:
          nonzero = np.ones(np.shape(y[:, task]), dtype=bool)
        # Set mean and std of present elements
        mean = np.mean(task_data[nonzero])
        std = np.std(task_data[nonzero])
        task_data[nonzero] = task_data[nonzero] - mean
        # Set standard deviation to one
        if std == 0.:
          print("Variance normalization skipped for task %d due to 0 stdev" % task)
          continue
        task_data[nonzero] = task_data[nonzero] / std
      else:
        raise ValueError("Unsupported Output transform")
  return y

def balance_positives(y, W):
  """Ensure that positive and negative examples have equal weight."""
  n_samples, n_targets = np.shape(y)
  for target_ind in range(n_targets):
    positive_inds, negative_inds = [], []
    to_next_target = False
    for sample_ind in range(n_samples):
      label = y[sample_ind, target_ind]
      if label == 1:
        positive_inds.append(sample_ind)
      elif label == 0:
        negative_inds.append(sample_ind)
      elif label == -1:  # Case of missing label
        continue
      else:
        warnings.warn("Labels must be 0/1 or -1 " +
                      "(missing data) for balance_positives target %d. " % target_ind +
                      "Continuing without balancing.")
        to_next_target = True
        break
    if to_next_target:
      continue
    n_positives, n_negatives = len(positive_inds), len(negative_inds)
    # TODO(rbharath): This results since the coarse train/test split doesn't
    # guarantee that the test set actually has any positives for targets. FIX
    # THIS BEFORE RELEASE!
    if n_positives == 0:
      pos_weight = 0
    else:
      pos_weight = float(n_negatives)/float(n_positives)
    W[positive_inds, target_ind] = pos_weight
    W[negative_inds, target_ind] = 1
  return W

#TODO(enf/rbharath): This is completely broken.

'''
def multitask_to_singletask(dataset):
  """Transforms a multitask dataset to a singletask dataset.

  Returns a dictionary which maps target names to datasets, where each
  dataset is itself a dict that maps identifiers to
  (fingerprint, scaffold, dict) tuples.

  Parameters
  ----------
  dataset: dict
    Dictionary of type produced by load_datasets
  """
  # Generate single-task data structures
  labels = dataset.itervalues().next()["labels"]
  sorted_targets = sorted(labels.keys())
  singletask_features = []
  singletask_labels = {target: [] for target in sorted_targets}
  # Populate the singletask datastructures
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

#TODO(enf/rbharath): Completly broken as well.
'''
def split_dataset(dataset, splittype, seed=None):
  """Split provided data using specified method."""
  if splittype == "random":
    train, test = train_test_random_split(dataset, seed=seed)
  elif splittype == "scaffold":
    train, test = train_test_scaffold_split(dataset)
  elif splittype == "specified":
    train, test = train_test_specified_split(dataset)
  else:
    raise ValueError("Improper splittype.")
  return train, test

def train_test_specified_split(dataset):
  """Split provided data due to splits in origin data."""
  train, test = {}, {}
  for mol_id, datapoint in dataset.iteritems():
    if "split" not in datapoint:
      raise ValueError("Missing required split information.")
    if datapoint["split"].lower() == "train":
      train[mol_id] = datapoint
    elif datapoint["split"].lower() == "test":
      test[mol_id] = datapoint
  return train, test

def train_test_random_split(dataset, frac_train=.8, seed=None):
  """Splits provided data into train/test splits randomly.

  Performs a random 80/20 split of the data into train/test. Returns two
  dictionaries

  Parameters
  ----------
  dataset: dict
    A dictionary of type produced by load_datasets.
  frac_train: float
    Proportion of data in train set.
  seed: int (optional)
    Seed to initialize np.random.
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
  """Splits provided data into train/test splits by scaffold.

  Groups the largest scaffolds into the train set until the size of the
  train set equals frac_train * len(dataset). Adds remaining scaffolds
  to test set. The idea is that the test set contains outlier scaffolds,
  and thus serves as a hard test of generalization capability for the
  model.

  Parameters
  ----------
  dataset: dict
    A dictionary of type produced by load_datasets.
  frac_train: float
    The fraction (between 0 and 1) of the data to use for train set.
  """
  scaffolds = scaffold_separate(dataset)
  train_size = frac_train * len(dataset)
  train, test = {}, {}
  for elements in scaffolds:
    # If adding this scaffold makes the train_set too big, add to test set.
    if len(train) + len(elements) > train_size:
      for elt in elements:
        test[elt] = dataset[elt]
    else:
      for elt in elements:
        train[elt] = dataset[elt]
  return train, test

def scaffold_separate(dataset):
  """Splits provided data by compound scaffolds.

  Returns a list of pairs (scaffold, [identifiers]), where each pair
  contains a scaffold and a list of all identifiers for compounds that
  share that scaffold. The list will be sorted in decreasing order of
  number of compounds.

  Parameters
  ----------
  dataset: dict
    A dictionary of type produced by load_datasets.
  """
  scaffolds = {}
  for mol_id in dataset:
    datapoint = dataset[mol_id]
    scaffold = datapoint["scaffold"]
    if scaffold not in scaffolds:
      scaffolds[scaffold] = [mol_id]
    else:
      scaffolds[scaffold].append(mol_id)
  # Sort from largest to smallest scaffold sets
  return [elt for (scaffold, elt) in sorted(scaffolds.items(), key=lambda x: -len(x[1]))]
'''