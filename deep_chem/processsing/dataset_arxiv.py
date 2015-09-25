"""
Code for processing the Google vs-datasets.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import os
import numpy as np
import gzip
import cPickle as pickle
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

def summarize_distribution(y):
  """Analyzes regression dataset.

  Parameters
  ----------
  y: np.ndarray 
    A 1D numpy array containing distribution.
  """
  mean = np.mean(y)
  std = np.std(y)
  minval = np.amin(y)
  maxval = np.amax(y)
  hist = np.histogram(y)
  print "Mean: %f" % mean
  print "Std: %f" % std
  print "Min: %f" % minval
  print "Max: %f" % maxval
  print "Histogram: "
  print hist

def load_descriptors(paths, descriptor_dir_name="descriptors"):
  """Load dataset descriptors and return.

  Returns a dictionary that maps mol_id to descriptors. (Descriptors are
  taken from rdkit.Chem and consist of basic physiochemical property
  values.

  Parameters
  ----------
  paths: list
    List of strings.
  """
  descriptor_dict = {}
  for dataset_path in paths:
    pickle_dir = os.path.join(dataset_path, descriptor_dir_name)
    for pickle_file in os.listdir(pickle_dir):
      with gzip.open(os.path.join(pickle_dir, pickle_file), "rb") as f:
        contents = pickle.load(f)
        all_smiles, descriptors = (contents["smiles"], contents["features"])
        for mol in range(len(all_smiles)):
          n_descriptors = len(descriptors[mol])
          bad_sets = [1, 28, 55, 68, 71, 72, 81, 83, 84, 85, 86, 87, 88,
          123, 124, 129, 137, 142, 143, 144, 146, 148, 149, 150, 156, 157,
          159, 160, 161, 164, 169, 170, 171, 173, 178, 179, 183, 190, 192]
          descriptor_dict[all_smiles[mol]] = np.array(
              [descriptors[mol][index] for index in range(n_descriptors) if
                  index not in bad_sets])
  return descriptor_dict

def load_molecules(paths, dir_name="circular-scaffold-smiles"):
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
    pickle_dir = os.path.join(dataset_path, dir_name)
    for pickle_file in os.listdir(pickle_dir):
      with gzip.open(os.path.join(pickle_dir, pickle_file), "rb") as f:
        contents = pickle.load(f)
        smiles, fingerprints, scaffolds, mol_ids = (
            contents["smiles"], contents["features"],
            contents["scaffolds"], contents["mol_id"])
        for mol in range(len(contents["smiles"])):
          molecules[smiles[mol]] = {"fingerprint": fingerprints[mol],
                                    "scaffold": scaffolds[mol],
                                    "mol_id": mol_ids[mol]}
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

def load_assays(paths, target_dir_name="targets"):
  """Load regression dataset labels from assays.

  Returns a dictionary that maps smiles strings to label vectors.

  TODO(rbharath): Simplify this function to only support the new pickle format.

  Parameters
  ----------
  paths: list 
    List of paths to base directory.
  target_dir_name: string
    Name of subdirectory containing assay data.
  """
  datapoints = {}
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
        if "potency" in contents:
          items = zip(contents["smiles"], contents["potency"])
        elif "targets" in contents:
          items = zip(contents["smiles"], contents["targets"])
        else:
          raise ValueError("Must contain either potency or targets field.")
        for mol, potency in items:
          # TODO(rbharath): Get a less kludgey answer
          # TODO(rbharath): There is some amount of duplicate collisions
          # due to choice of smiles generation. Look into this more
          # carefully and see if the underlying issues are fundamental..
          try:
            if potency is None or np.isnan(potency):
              continue
          except TypeError:
            continue
          if mol not in datapoints:
            datapoints[mol] = {}
            # Ensure that each target has some entry in dict.
            for name in target_names:
              # Set all targets to invalid for now.
              datapoints[mol][name] = -1
          datapoints[mol][target_name] = potency 
  return datapoints

def compare_all_datasets():
  """Compare all datasets in our collection.

  TODO(rbharath): Make this actually robust.
  """
  muv_path = "/home/rbharath/vs-datasets/muv"
  pcba_path = "/home/rbharath/vs-datasets/pcba"
  dude_path = "/home/rbharath/vs-datasets/dude"
  pfizer_path = "/home/rbharath/private-datasets/pfizer"
  muv_data = load_datasets([muv_path])
  pcba_data = load_datasets([pcba_path])
  dude_data = load_datasets([dude_path])
  pfizer_data = load_datasets([pfizer_path])
  print "----------------------"
  compare_datasets("muv", muv_data, "pcba", pcba_data)
  print "----------------------"
  compare_datasets("pfizer", pfizer_data, "muv", muv_data)
  print "----------------------"
  compare_datasets("pfizer", pfizer_data, "pcba", pcba_data)
  print "----------------------"
  compare_datasets("muv", muv_data, "dude", dude_data)
  print "----------------------"
  compare_datasets("pcba", pcba_data, "dude", dude_data)
  print "----------------------"
  compare_datasets("pfizer", pfizer_data, "dude", dude_data)

def compare_datasets(first_name, first, second_name, second):
  """Counts the overlap between two provided datasets.

  Parameters
  ----------
  first_name: string
    Name of first dataset
  first: dict
    Data dictionary generated by load_datasets.
  second_name: string
    Name of second dataset
  second: dict
    Data dictionary generated by load_datasets.
  """
  first_scaffolds = set()
  for key in first:
    _, scaffold, _ = first[key]
    first_scaffolds.add(scaffold)
  print "%d molecules in %s dataset" % (len(first), first_name)
  print "%d scaffolds in %s dataset" % (len(first_scaffolds), first_name)
  second_scaffolds = set()
  for key in second:
    _, scaffold, _ = second[key]
    second_scaffolds.add(scaffold)
  print "%d molecules in %s dataset" % (len(second), second_name)
  print "%d scaffolds in %s dataset" % (len(second_scaffolds), second_name)
  common_scaffolds = first_scaffolds.intersection(second_scaffolds)
  print "%d scaffolds in both" % len(common_scaffolds)

def load_datasets(paths, datatype="vs", **load_args):
  """Dispatches to correct loader depending on type of data."""
  if datatype == "vs":
    return load_vs_datasets(paths, **load_args)
  elif datatype == "pdbbind":
    return load_pdbbind_datasets(paths, **load_args)
  else:
    raise ValueError("Unsupported datatype.")

def load_pdbbind_datasets(pdbbind_paths):
  """Load pdbbind datasets.

  Parameters
  ----------
  pdbbind_path: list 
    List of Pdbbind data files.
  """
  data = []
  for pdbbind_path in pdbbind_paths:
    with open(pdbbind_path, "rb") as csvfile:
      reader = csv.reader(csvfile)
      for row_ind, row in enumerate(reader):
        if row_ind == 0:
          continue
        data.append({
          "label": row[0],
          "features": row[1],
        })
  df = pd.DataFrame(data)
  print df.shape
  return df

def load_vs_datasets(paths, target_dir_name="targets",
    fingerprint_dir_name="circular-scaffold-smiles",
    descriptor_dir_name="descriptors",
    add_descriptors=False):
  """Load both labels and fingerprints.

  Returns a dictionary that maps smiles to pairs of (fingerprint, labels)
  where labels is itself a dict that maps target-names to labels.

  Parameters
  ----------
  paths: string or list
    Paths to base directory.
  """
  data = {}
  molecules = load_molecules(paths, fingerprint_dir_name)
  labels = load_assays(paths, target_dir_name)
  if add_descriptors:
    descriptors = load_descriptors(paths, descriptor_dir_name)
  # TODO(rbharath): Why are there fewer descriptors than labels at times?
  # What accounts for the descrepency. Please investigate.
  for ind, smiles in enumerate(molecules):
    if smiles not in labels or (add_descriptors and smiles not in descriptors):
      continue
    mol = molecules[smiles]
    if add_descriptors:
      data[smiles] = {"fingerprint": mol["fingerprint"],
                      "scaffold": mol["scaffold"],
                      "labels": labels[smiles],
                      "descriptors": descriptors[smiles]}
    else:
      data[smiles] = {"fingerprint": mol["fingerprint"],
                      "scaffold": mol["scaffold"],
                      "labels": labels[smiles]}
  return data

def get_default_descriptor_transforms():
  """Provides default descriptor transforms for rdkit descriptors."""
  # TODO(rbharath): Remove these magic numbers 
  n_descriptors = 196 - 39
  for desc in range(n_descriptors):
    desc_transforms[desc] = ["normalize"]
  return desc_transforms

def get_default_task_types_and_transforms(dataset_specs):
  """Provides default task transforms for provided datasets.
  
  Parameters
  ----------
  dataset_specs: dict
    Maps name of datasets to filepath.
  """
  task_types, task_transforms = {}, {}
  for name, path in dataset_specs.itervalues():
    targets = get_target_names([path])
    if name == "muv" or name == "dude" or name == "pcba":
      for target in targets:
        task_types[target] = "classification"
        task_transforms[target] = []
    elif name == "pfizer":
      for target in targets:
        task_types[target] = "regression"
        task_transforms[target] = ["log", "normalize"]
    elif name == "pdbbind":

  return task_types, task_transforms

def transform_outputs(dataset, task_transforms, desc_transforms={},
    add_descriptors=False):
  """Tranform the provided outputs

  Parameters
  ----------
  dataset: dict 
    A dictionary of type produced by load_datasets. 
  task_transforms: dict 
    dict mapping target names to list of label transforms. Each list
    element must be "1+max-val", "log", "normalize". The transformations are
    performed in the order specified. An empty list
    corresponds to no transformations. Only for regression outputs.
  desc_transforms: dict
    dict mapping descriptor number to transform. Each transform must be
    either None, "log", "normalize", or "log-normalize"
  add_descriptors: bool
    Add descriptor prediction as extra task.
  """
  X, y, W = dataset_to_numpy(dataset, add_descriptors=add_descriptors)
  sorted_targets = sorted(task_transforms.keys())
  if add_descriptors:
    sorted_descriptors = sorted(desc_transforms.keys())
    endpoints = sorted_targets + sorted_descriptors
  else:
    endpoints = sorted_targets
  transforms = task_transforms.copy()
  if add_descriptors:
    transforms.update(desc_transforms)
  for task, target in enumerate(endpoints):
    task_transforms = transforms[target]
    print "Task %d has NaNs?" % task
    print np.any(np.isnan(y[:, task]))
    print "Task %d data" % task
    print y[:, task]
    print "Task %d distribution" % task
    summarize_distribution(y[:, task])
    for task_transform in task_transforms:
      if task_transform == "log":
        y[:, task] = np.log(y[:, task])
      elif task_transform == "1+max-val":
        maxval = np.amax(y[:, task])
        y[:, task] = 1 + maxval - y[:, task]
      elif task_transform == "normalize":
        task_data = y[:, task]
        if task < len(sorted_targets):
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
        raise ValueError("Task tranform must be 1+max-val, log, or normalize")
    print "Post-transform task %d distribution" % task
    summarize_distribution(y[:, task])
  return X, y, W

def load_and_transform_dataset(paths, task_transforms, desc_transforms={},
    labels_endpoint="labels", descriptors_endpoint="descriptors",
    add_descriptors=False):
  """Transform data labels as specified

  Parameters
  ----------
  paths: list 
    List of paths to Google vs datasets. 
  task_transforms: dict 
    dict mapping target names to list of label transforms. Each list
    element must be "max-val", "log", "normalize". The transformations are
    performed in the order specified. An empty list
    corresponds to no transformations. Only for regression outputs.
  desc_transforms: dict
    dict mapping descriptor number to transform. Each transform must be
    either None, "log", "normalize", or "log-normalize"
  add_descriptors: bool
    Add descriptor prediction as extra task.
  """
  dataset = load_datasets(paths, add_descriptors=add_descriptors)
  X, y, W = transform_outputs(dataset, task_transforms,
      desc_transforms=desc_transforms, add_descriptors=add_descriptors)
  trans_data = {}
  sorted_smiles = sorted(dataset.keys())
  sorted_targets = sorted(task_transforms.keys())
  for s_index, smiles in enumerate(sorted_smiles):
    datapoint = dataset[smiles]
    labels = {}
    for t_index, target in enumerate(sorted_targets):
      if W[s_index][t_index] == 0:
        labels[target] = -1
      else:
        labels[target] = y[s_index][t_index]
    datapoint[labels_endpoint] = labels
    if add_descriptors:
      # All non-target endpoints are descriptors
      datapoint[descriptors_endpoint] = y[s_index][len(sorted_targets):]
    trans_data[smiles] = datapoint 
  return trans_data

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
  singletask = {}
  for target in sorted_targets:
    singletask[target] = {} 
  # Populate the singletask datastructures
  sorted_smiles = sorted(dataset.keys())
  for index, smiles in enumerate(sorted_smiles):
    datapoint = dataset[smiles]
    labels = datapoint["labels"]
    for t_ind, target in enumerate(sorted_targets):
      if labels[target] == -1:
        continue
      else:
        datapoint_copy = datapoint.copy()
        datapoint_copy["labels"] = {target: labels[target]}
        singletask[target][smiles] = datapoint_copy 
  return singletask

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

def dataset_to_numpy(dataset, feature_endpoint="fingerprint",
    labels_endpoint="labels", descriptors_endpoint="descriptors",
    desc_weight=.5, add_descriptors=False):
  """Transforms a loaded dataset into numpy arrays (X, y).

  Transforms provided dict into feature matrix X (of dimensions [n_samples,
  n_features]) and label matrix y (of dimensions [n_samples,
  n_targets+n_desc]), where n_targets is the number of assays in the
  provided datset and n_desc is the number of computed descriptors we'd
  like to predict.

  Note that this function transforms missing data into negative examples
  (this is relatively safe since the ratio of positive to negative examples
  is on the order 1/100)
  
  Parameters
  ----------
  dataset: dict 
    A dictionary of type produced by load_datasets. 
  add_descriptors: bool
    Add descriptor prediction as extra task.
  """
  n_samples = len(dataset.keys())
  sample_datapoint = dataset.itervalues().next()
  n_features = len(sample_datapoint[feature_endpoint])
  n_targets = len(sample_datapoint[labels_endpoint])
  X = np.zeros((n_samples, n_features))
  if add_descriptors:
    n_desc = len(sample_datapoint[descriptors_endpoint])
    y = np.zeros((n_samples, n_targets + n_desc))
    W = np.ones((n_samples, n_targets + n_desc))
  else:
    y = np.zeros((n_samples, n_targets))
    W = np.ones((n_samples, n_targets))
  sorted_smiles = sorted(dataset.keys())
  for index, smiles in enumerate(sorted_smiles):
    datapoint = dataset[smiles] 
    fingerprint, labels  = (datapoint[feature_endpoint],
        datapoint[labels_endpoint])
    if add_descriptors:
      descriptors = datapoint[descriptors_endpoint]
    X[index] = np.array(fingerprint)
    sorted_targets = sorted(labels.keys())
    # Set labels from measurements
    for t_ind, target in enumerate(sorted_targets):
      if labels[target] == -1:
        y[index][t_ind] = 0
        W[index][t_ind] = 0
      else:
        y[index][t_ind] = labels[target]
    if add_descriptors:
      # Set labels from descriptors
      y[index][n_targets:] = descriptors
      W[index][n_targets:] = desc_weight
  return X, y, W

def train_test_random_split(dataset, frac_train=.8, seed=None):
  """Splits provided data into train/test splits randomly.

  Performs a random 80/20 split of the data into train/test. Returns two
  dictionaries

  Parameters
  ----------
  dataset: dict 
    A dictionary of type produced by load_datasets. 
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
  for smiles in dataset:
    datapoint = dataset[smiles]
    scaffold = datapoint["scaffold"]
    if scaffold not in scaffolds:
      scaffolds[scaffold] = [smiles]
    else:
      scaffolds[scaffold].append(smiles)
  # Sort from largest to smallest scaffold sets 
  return sorted(scaffolds.items(), key=lambda x: -len(x[1]))

def model_predictions(test_set, model, n_targets, n_descriptors=0,
    add_descriptors=False, modeltype="sklearn"):
  """Obtains predictions of provided model on test_set.

  Returns a list of per-task predictions.

  TODO(rbharath): This function uses n_descriptors, n_targets instead of
  task_transforms, desc_transforms like everything else.

  Parameters
  ----------
  test_set: dict 
    A dictionary of type produced by load_datasets. Contains the test-set.
  model: model.
    A trained scikit-learn or keras model.
  n_targets: int
    Number of output targets
  n_descriptors: int
    Number of output descriptors
  modeltype: string
    Either sklearn, keras, or keras_multitask
  add_descriptors: bool
    Add descriptor prediction as extra task.
  """
  # Extract features for test set and make preds
  X, _, _ = dataset_to_numpy(test_set)
  if add_descriptors:
    n_outputs = n_targets + n_descriptors
  else:
    n_outputs = n_targets
  if modeltype == "sklearn":
    ypreds = model.predict_proba(X)
  elif modeltype == "keras":
    ypreds = model.predict(X)
  elif modeltype == "keras_multitask":
    predictions = model.predict({"input": X})
    ypreds = []
    for index in range(n_outputs):
      ypreds.append(predictions["task%d" % index])
  else:
    raise ValueError("Improper modeltype.")
  # Handle the edge case for singletask. 
  if type(ypreds) != list:
    ypreds = [ypreds]
  return ypreds
  
def eval_model(test_set, model, task_types, desc_transforms={}, modeltype="sklearn",
    add_descriptors=False):
  """Evaluates the provided model on the test-set.

  Returns a dict which maps target-names to pairs of np.ndarrays (ytrue,
  yscore) of true labels vs. predict

  TODO(rbharath): This function is too complex. Refactor and simplify.
  TODO(rbharath): The handling of add_descriptors for semi-supervised learning
  is horrible. Refactor.

  Parameters
  ----------
  test_set: dict 
    A dictionary of type produced by load_datasets. Contains the test-set.
  model: model.
    A trained scikit-learn or keras model.
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  desc_transforms: dict
    dict mapping descriptor number to transform. Each transform must be
    either None, "log", "normalize", or "log-normalize"
  modeltype: string
    Either sklearn, keras, or keras_multitask
  add_descriptors: bool
    Add descriptor prediction as extra task.
  """
  sorted_targets = sorted(task_types.keys())
  if add_descriptors:
    sorted_descriptors = sorted(desc_transforms.keys())
    endpoints = sorted_targets + sorted_descriptors
    local_task_types = task_types.copy()
    for desc in desc_transforms:
      local_task_types[desc] = "regression"
  else:
    local_task_types = task_types.copy()
    endpoints = sorted_targets
  ypreds = model_predictions(test_set, model, len(sorted_targets),
      n_descriptors=len(desc_transforms), modeltype=modeltype,
      add_descriptors=add_descriptors)
  results = {}
  for target in endpoints:
    results[target] = ([], [])  # (ytrue, yscore)
  # Iterate through test set data points.
  sorted_smiles = sorted(test_set.keys())
  for index, smiles in enumerate(sorted_smiles):
    datapoint = test_set[smiles]
    labels = datapoint["labels"]
    for t_ind, target in enumerate(endpoints):
      task_type = local_task_types[target]
      if target in sorted_targets and labels[target] == -1:
        continue
      else:
        ytrue, yscore = results[target]
        if task_type == "classification":
          if labels[target] == 0:
            ytrue.append(0)
          elif labels[target] == 1:
            ytrue.append(1)
          else:
            raise ValueError("Labels must be 0/1.")
        elif target in sorted_targets and task_type == "regression":
          ytrue.append(labels[target])
        elif target not in sorted_targets and task_type == "regression":
          descriptors = datapoint["descriptors"]
          # The "target" for descriptors is simply the index in the
          # descriptor vector.
          ytrue.append(descriptors[int(target)])
        else:
          raise ValueError("task_type must be classification or regression.")
        yscore.append(ypreds[t_ind][index])
  for target in endpoints:
    ytrue, yscore = results[target]
    results[target] = (np.array(ytrue), np.array(yscore))
  return results

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


def compute_roc_auc_scores(results, task_types):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_model which maps target-names to
    pairs of lists (ytrue, yscore).
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for target in results:
    if task_types[target] != "classification":
      continue
    ytrue, yscore = results[target]
    sample_weights = labels_to_weights(ytrue)
    print "np.shape(ytrue)"
    print np.shape(ytrue)
    print "np.shape(yscore)"
    print np.shape(yscore)
    score = roc_auc_score(ytrue, yscore[:,1], sample_weight=sample_weights)
    #score = roc_auc_score(ytrue, yscore, sample_weight=sample_weights)
    print "Target %s: AUC %f" % (target, score)
    scores[target] = score
  return scores

def compute_r2_scores(results, task_types):
  """Transforms the results dict into R^2 values and prints them.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_regression_model which maps target-names to
    pairs of lists (ytrue, yscore).
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for target in results:
    if task_types[target] != "regression":
      continue
    ytrue, yscore = results[target]
    score = r2_score(ytrue, yscore)
    print "Target %s: R^2 %f" % (target, score)
    scores[target] = score
  return scores

def compute_rms_scores(results, task_types):
  """Transforms the results dict into RMS values and prints them.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_regression_model which maps target-names to
    pairs of lists (ytrue, yscore).
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for target in results:
    if task_types[target] != "regression":
      continue
    ytrue, yscore = results[target]
    rms = np.sqrt(mean_squared_error(ytrue, yscore))
    print "Target %s: RMS %f" % (target, rms)
    scores[target] = rms 
  return scores
