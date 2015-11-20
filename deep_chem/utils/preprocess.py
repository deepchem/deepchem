"""
Utility functions to preprocess datasets before building models.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import numpy as np
import warnings
from deep_chem.utils.analysis import summarize_distribution

def to_arrays(train, test, dtype=float):
  """Turns train/test into numpy array."""
  train_ids, X_train, y_train, W_train = dataset_to_numpy(train, dtype=dtype)
  test_ids, X_test, y_test, W_test = dataset_to_numpy(test, dtype=dtype)
  return (train_ids, X_train, y_train, W_train), (test_ids, X_test, y_test, W_test)

def transform_inputs(X, input_transforms):
  """Transform the input feature data."""
  # Short-circuit to handle difficulties with strings.
  if not input_transforms:
    return X
  # Copy X up front to have non-destructive updates.
  X = np.copy(X)
  if len(np.shape(X)) == 2:
    (n_samples, n_features) = np.shape(X)
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
  (_, n_targets) = np.shape(y)
  for task in range(n_targets):
    for output_transform in output_transforms:
      if output_transform == "log":
        y[:, task] = np.log(y[:, task])
      elif output_transform == "normalize":
        task_data = y[:, task]
        if task < n_targets:
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
          print "Variance normalization skipped for task %d due to 0 stdev" % task
          continue
        task_data[nonzero] = task_data[nonzero] / std
      else:
        raise ValueError("Unsupported Output transform")
  return y

def to_one_hot(y):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape [n_samples, 2] (assuming binary labels).

  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, 2))
  for index, val in enumerate(y):
    if val == 0:
      y_hot[index] = np.array([1, 0])
    elif val == 1:
      y_hot[index] = np.array([0, 1])
  return y_hot

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
    print "For target %d, n_positives: %d, n_negatives: %d" % (
        target_ind, n_positives, n_negatives)
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

def dataset_to_numpy(dataset, weight_positives=True, dtype=float):
  """Transforms a set of tensor data into numpy arrays (X, y)"""
  n_samples = len(dataset.keys())
  sample_datapoint = dataset.itervalues().next()
  feature_shape = np.shape(sample_datapoint["fingerprint"])
  n_targets = len(sample_datapoint["labels"])
  X = np.squeeze(np.zeros((n_samples,) + feature_shape + (n_targets,), dtype=dtype))
  y = np.zeros((n_samples, n_targets))
  W = np.ones((n_samples, n_targets))
  sorted_ids = sorted(dataset.keys())
  for id_ind, id in enumerate(sorted_ids):
    datapoint = dataset[id]
    fingerprint, labels = (datapoint["fingerprint"],
      datapoint["labels"])
    X[id_ind] = np.reshape(fingerprint, np.shape(X[id_ind]))
    sorted_targets = sorted(labels.keys())
    # Set labels from measurements
    for target_ind, target in enumerate(sorted_targets):
      if labels[target] == -1:
        y[id_ind][target_ind] = -1
        W[id_ind][target_ind] = 0
      else:
        y[id_ind][target_ind] = labels[target]
  if weight_positives:
    W = balance_positives(y, W)
  return (sorted_ids, X, y, W)

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
  singletask = {target: {} for target in sorted_targets}
  # Populate the singletask datastructures
  sorted_ids = sorted(dataset.keys())
  for index, id in enumerate(sorted_ids):
    datapoint = dataset[id]
    labels = datapoint["labels"]
    for t_ind, target in enumerate(sorted_targets):
      if labels[target] == -1:
        continue
      else:
        datapoint_copy = datapoint.copy()
        datapoint_copy["labels"] = {target: labels[target]}
        singletask[target][id] = datapoint_copy 
  return singletask

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
  for id, datapoint in dataset.iteritems():
    if "split" not in datapoint:
      raise ValueError("Missing required split information.")
    if datapoint["split"].lower() == "train":
      train[id] = datapoint
    elif datapoint["split"].lower() == "test":
      test[id] = datapoint
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
  train, test= {}, {}
  for scaffold, elements in scaffolds:
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
  for id in dataset:
    datapoint = dataset[id]
    scaffold = datapoint["scaffold"]
    if scaffold not in scaffolds:
      scaffolds[scaffold] = [id]
    else:
      scaffolds[scaffold].append(id)
  # Sort from largest to smallest scaffold sets 
  return sorted(scaffolds.items(), key=lambda x: -len(x[1]))

def labels_to_weights(ytrue):
  """Uses the true labels to compute and output sample weights.

  Parameters
  ----------
  ytrue: list or np.ndarray
    True labels.
  """
  n_total = np.shape(ytrue)[0]
  n_positives = np.sum(ytrue)
  n_negatives = n_total - n_positives
  pos_weight = np.floor(n_negatives/n_positives)

  sample_weights = np.zeros(np.shape(ytrue)[0])
  for ind, entry in enumerate(ytrue):
    if entry == 0:  # negative
      sample_weights[ind] = 1
    elif entry == 1:  # positive
      sample_weights[ind] = pos_weight
    else:
      raise ValueError("ytrue can only contain 0s or 1s.")
  return sample_weights
