"""
Contains wrapper class for datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import load_from_disk
from deepchem.featurizers.featurize import FeaturizedSamples

# TODO(rbharath): The semantics of this class are very difficult to debug.
# Multiple transformations of the data are performed on disk, and computations
# of mean/std are spread across multiple functions for efficiency. Some
# refactoring needs to happen here.
class Dataset(object):
  """
  Wrapper class for dataset transformed into X, y, w numpy ndarrays.
  """
  def __init__(self, data_dir=None, tasks=[], samples=None, featurizers=None, 
               use_user_specified_features=False):
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    self.data_dir = data_dir

    if featurizers is not None:
      feature_types = [featurizer.__class__.__name__ for featurizer in featurizers]
    else:
      feature_types = None

    if use_user_specified_features:
      feature_types = ["user-specified-features"]

    if samples is not None and feature_types is not None:
      if not isinstance(feature_types, list):
        raise ValueError("feature_types must be a list or None.")

      write_dataset_single_partial = partial(
          write_dataset_single, data_dir=self.data_dir,
          feature_types=feature_types, tasks=tasks)

      metadata_rows = []
      # TODO(rbharath): Still a bit of information leakage.
      for df_file, df in zip(samples.dataset_files, samples.iterdataframes()):
        retval = write_dataset_single_partial((df_file, df))
        if retval is not None:
          metadata_rows.append(retval)

      self.metadata_df = pd.DataFrame(
          metadata_rows,
          columns=('df_file', 'task_names', 'ids',
                   'X', 'X-transformed', 'y', 'y-transformed',
                   'w',
                   'X_sums', 'X_sum_squares', 'X_n',
                   'y_sums', 'y_sum_squares', 'y_n'))
      self.save_to_disk()
    else:
      if os.path.exists(self._get_metadata_filename()):
        self.metadata_df = load_from_disk(self._get_metadata_filename())
      else:
        raise ValueError("No metadata found.")

  def save_to_disk(self):
    """Save dataset to disk."""
    save_to_disk(
        self.metadata_df, self._get_metadata_filename())

  def get_task_names(self):
    """
    Gets learning tasks associated with this dataset.
    """
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    return self.metadata_df.iterrows().next()[1]['task_names']

  def get_data_shape(self):
    """
    Gets array shape of datapoints in this dataset.
    """
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_X = load_from_disk(self.metadata_df.iterrows().next()[1]['X'])[0]
    return np.shape(sample_X)

  def _get_metadata_filename(self):
    """
    Get standard location for metadata file.
    """
    metadata_filename = os.path.join(self.data_dir, "metadata.joblib")
    return metadata_filename

  def get_number_shards(self):
    """
    Returns the number of shards for this dataset.
    """
    return self.metadata_df.shape[0]

  # TODO(rbharath): There is a dangerous mixup in semantics. If itershards() is
  # called without calling transform(), it will explode. Maybe have a separate
  # initialization function to avoid this problem.
  def itershards(self):
    """
    Iterates over all shards in dataset.
    """
    for _, row in self.metadata_df.iterrows():
      X = load_from_disk(row['X-transformed'])
      y = load_from_disk(row['y-transformed'])
      w = load_from_disk(row['w'])
      ids = load_from_disk(row['ids'])
      yield (X, y, w, ids)

  def __len__(self):
    """
    Finds number of elements in dataset.
    """
    total = 0
    for _, row in self.metadata_df.iterrows():
      y = load_from_disk(row['y-transformed'])
      total += len(y)
    return total

  def get_label_means(self):
    """Return pandas series of label means."""
    return self.metadata_df["y_means"]

  def get_label_stds(self):
    """Return pandas series of label stds."""
    return self.metadata_df["y_stds"]

  def compute_statistics(self):
    """Computes statistics of this dataset"""
    df = self.metadata_df
    X_means, X_stds, y_means, y_stds = compute_mean_and_std(df)
    return X_means, X_stds, y_means, y_stds
    
def compute_sums_and_nb_sample(tensor, W=None):
  """
  Computes sums, squared sums of tensor along axis 0.

  If W is specified, only nonzero weight entries of tensor are used.
  """
  if len(np.shape(tensor)) == 1:
    tensor = np.reshape(tensor, (len(tensor), 1))
  if W is not None and len(np.shape(W)) == 1:
    W = np.reshape(W, (len(W), 1))
  if W is None:
    sums = np.sum(tensor, axis=0)
    sum_squares = np.sum(np.square(tensor), axis=0)
    nb_sample = np.shape(tensor)[0]
  else:
    nb_task = np.shape(tensor)[1]
    sums = np.zeros(nb_task)
    sum_squares = np.zeros(nb_task)
    nb_sample = np.zeros(nb_task)
    for task in range(nb_task):
      y_task = tensor[:, task]
      W_task = W[:, task]
      nonzero_indices = np.nonzero(W_task)[0]
      y_task_nonzero = y_task[nonzero_indices]
      sums[task] = np.sum(y_task_nonzero)
      sum_squares[task] = np.dot(y_task_nonzero, y_task_nonzero)
      nb_sample[task] = np.shape(y_task_nonzero)[0]
  return (sums, sum_squares, nb_sample)

# The following are all associated with Dataset, but are separate functions to
# make it easy to use multiprocessing.

def write_dataset_single(val, data_dir, feature_types, tasks):
  """Writes files for single row (X, y, w, X-transformed, ...) to disk."""
  (df_file, df) = val
  # TODO(rbharath): This is a hack. clean up.
  if not len(df):
    return None
  task_names = sorted(tasks)
  ids, X, y, w = _df_to_numpy(df, feature_types, tasks)
  X_sums, X_sum_squares, X_n = compute_sums_and_nb_sample(X)
  y_sums, y_sum_squares, y_n = compute_sums_and_nb_sample(y, w)

  basename = os.path.splitext(os.path.basename(df_file))[0]
  out_X = os.path.join(data_dir, "%s-X.joblib" % basename)
  out_X_transformed = os.path.join(data_dir, "%s-X-transformed.joblib" % basename)
  out_y = os.path.join(data_dir, "%s-y.joblib" % basename)
  out_y_transformed = os.path.join(data_dir, "%s-y-transformed.joblib" % basename)
  out_w = os.path.join(data_dir, "%s-w.joblib" % basename)
  out_ids = os.path.join(data_dir, "%s-ids.joblib" % basename)

  save_to_disk(X, out_X)
  save_to_disk(y, out_y)
  # Write X, y as transformed versions
  save_to_disk(X, out_X_transformed)
  save_to_disk(y, out_y_transformed)
  save_to_disk(w, out_w)
  save_to_disk(ids, out_ids)
  # TODO(rbharath): Should X be saved to out_X_transformed as well? Since
  # itershards expects to loop over X-transformed? (Ditto for y/w)
  return([df_file, task_names, out_ids, out_X, out_X_transformed, out_y,
          out_y_transformed, out_w,
          X_sums, X_sum_squares, X_n,
          y_sums, y_sum_squares, y_n])

def _df_to_numpy(df, feature_types, tasks):
  """Transforms a featurized dataset df into standard set of numpy arrays"""
  if not set(feature_types).issubset(df.keys()):
    raise ValueError(
        "Featurized data does not support requested feature_types.")
  # perform common train/test split across all tasks
  n_samples = df.shape[0]
  sorted_tasks = sorted(tasks)
  n_tasks = len(sorted_tasks)
  y = df[sorted_tasks].values
  y = np.reshape(y, (n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  tensors = []
  for ind , datapoint in df.iterrows():
    feature_list = []
    for feature_type in feature_types:
      feature_list.append(datapoint[feature_type])
    # TODO(rbharath): Total hack. Fix before merge!!!
    try:
      features = np.squeeze(np.concatenate(feature_list))
      for ind, val in enumerate(features):
        if features[ind] == "":
          features[ind] = 0.
      features = features.astype(float)
    except ValueError:
      y[ind] = ""
      continue
    tensors.append(features)
  x = np.stack(tensors)
  sorted_ids = df["mol_id"]

  # Set missing data to have weight zero
  missing = (y.astype(object) == "")
  y[missing] = 0.
  w[missing] = 0.

  return sorted_ids, x.astype(float), y.astype(float), w.astype(float)

def compute_mean_and_std(df):
  """
  Compute means/stds of X/y from sums/sum_squares of tensors.
  """
  X_sums, X_sum_squares, X_n = (list(df['X_sums']),
                                list(df['X_sum_squares']),
                                list(df['X_n']))
  # Note that X_n is a list of floats
  n = float(np.sum(X_n))
  X_sums = np.vstack(X_sums)
  X_sum_squares = np.vstack(X_sum_squares)
  overall_X_sums = np.sum(X_sums, axis=0)
  overall_X_means = overall_X_sums / n
  overall_X_sum_squares = np.sum(X_sum_squares, axis=0)

  X_vars = (overall_X_sum_squares - np.square(overall_X_sums)/n)/(n)

  y_sums, y_sum_squares, y_n = (list(df['y_sums']),
                                list(df['y_sum_squares']),
                                list(df['y_n']))
  # Note y_n is a list of arrays of shape (n_tasks,)
  y_n = np.sum(y_n, axis=0)
  y_sums = np.vstack(y_sums)
  y_sum_squares = np.vstack(y_sum_squares)
  y_means = np.sum(y_sums, axis=0)/y_n
  y_vars = np.sum(y_sum_squares, axis=0)/y_n - np.square(y_means)
  return overall_X_means, np.sqrt(X_vars), y_means, np.sqrt(y_vars)
