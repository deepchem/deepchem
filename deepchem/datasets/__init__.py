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
from deepchem.utils.save import log

# TODO(rbharath): The semantics of this class are very difficult to debug.
# Multiple transformations of the data are performed on disk, and computations
# of mean/std are spread across multiple functions for efficiency. Some
# refactoring needs to happen here.
class Dataset(object):
  """
  Wrapper class for dataset transformed into X, y, w numpy ndarrays.
  """
  def __init__(self, data_dir=None, tasks=[], samples=None, featurizers=None, 
               use_user_specified_features=False,
               raw_data=None, verbosity=None, reload=False):
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    self.data_dir = data_dir
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity

    if featurizers is not None:
      feature_types = [featurizer.__class__.__name__ for featurizer in featurizers]
    else:
      feature_types = None

    if not reload or not os.path.exists(self._get_metadata_filename()):
      log("About to start initializing dataset", self.verbosity)
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
        for ind, (df_file, df) in enumerate(
            zip(samples.dataset_files, samples.iterdataframes())):
          log("Writing data from file %s, number %d/%d"
              % (df_file, ind, len(samples.dataset_files)), self.verbosity)
          retval = write_dataset_single_partial((df_file, df))
          if retval is not None:
            metadata_rows.append(retval)

        self.metadata_df = pd.DataFrame(
            metadata_rows,
            columns=('df_file', 'task_names', 'ids',
                     'X', 'X-transformed', 'y', 'y-transformed',
                     'w', 'w-transformed',
                     'X_sums', 'X_sum_squares', 'X_n',
                     'y_sums', 'y_sum_squares', 'y_n'))
        self.save_to_disk()
      elif raw_data is not None:
        metadata_rows = []
        metadata_rows.append(
            write_dataset_single(val=None, data_dir=self.data_dir, raw_data=raw_data,
                                 basename="data"))
        self.metadata_df = pd.DataFrame(
            metadata_rows,
            columns=('df_file', 'task_names', 'ids',
                     'X', 'X-transformed', 'y', 'y-transformed',
                     'w', 'w-transformed',
                     'X_sums', 'X_sum_squares', 'X_n',
                     'y_sums', 'y_sum_squares', 'y_n'))
        self.save_to_disk()
      #if samples is None and feature_types is not None:  
      else:
        # Create an empty metadata dataframe to be filled at a later time
        basename = "metadata"
        df_file = "metadata.joblib"
        out_X = os.path.join(data_dir, "%s-X.joblib" % basename)
        out_X_transformed = os.path.join(data_dir, "%s-X-transformed.joblib" % basename)
        out_X_sums = os.path.join(data_dir, "%s-X_sums.joblib" % basename)
        out_X_sum_squares = os.path.join(data_dir, "%s-X_sum_squares.joblib" % basename)
        out_X_n = os.path.join(data_dir, "%s-X_n.joblib" % basename)
        out_y = os.path.join(data_dir, "%s-y.joblib" % basename)
        out_y_transformed = os.path.join(data_dir, "%s-y-transformed.joblib" % basename)
        out_y_sums = os.path.join(data_dir, "%s-y_sums.joblib" % basename)
        out_y_sum_squares = os.path.join(data_dir, "%s-y_sum_squares.joblib" % basename)
        out_y_n = os.path.join(data_dir, "%s-y_n.joblib" % basename)
        out_w = os.path.join(data_dir, "%s-w.joblib" % basename)
        out_w_transformed = os.path.join(data_dir, "%s-w-transformed.joblib" % basename)
        out_ids = os.path.join(data_dir, "%s-ids.joblib" % basename)

        metadata_rows = []
        retval = ([df_file, tasks, out_ids,
                   out_X, out_X_transformed,
                   out_y, out_y_transformed,
                   out_w, out_w_transformed,
                   out_X_sums, out_X_sum_squares, out_X_n,
                   out_y_sums, out_y_sum_squares, out_y_n])
        metadata_rows.append(retval)

        self.metadata_df = pd.DataFrame(
            metadata_rows,
            columns=('df_file','task_names', 'ids',
                     'X', 'X-transformed', 'y', 'y-transformed',
                     'w', 'w-transformed',
                     'X_sums', 'X_sum_squares', 'X_n',
                     'y_sums', 'y_sum_squares', 'y_n'))
        self.save_to_disk()

    else:
      log("Loading pre-existing metadata file.", self.verbosity)
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
    sample_X = load_from_disk(
        self.metadata_df.iterrows().next()[1]['X-transformed'])[0]
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

  def itershards(self):
    """
    Iterates over all shards in dataset.

    Datasets are stored in sharded fashion on disk. Each call to next() for the
    generator defined by this function returns the data from a particular shard.
    The order of shards returned is guaranteed to remain fixed.
    """
    for _, row in self.metadata_df.iterrows():
      X = load_from_disk(row['X-transformed'])
      y = load_from_disk(row['y-transformed'])
      w = load_from_disk(row['w-transformed'])
      ids = load_from_disk(row['ids'])
      yield (X, y, w, ids)

  def iterbatches(self, batch_size=None, epoch=0):
    """
    Returns minibatches from dataset.
    """
    if batch_size == None:
      batch_size = len(self)
    for i, (X, y, w, ids) in enumerate(self.itershards()):
      nb_sample = np.shape(X)[0]
      interval_points = np.linspace(
          0, nb_sample, np.ceil(float(nb_sample)/batch_size)+1, dtype=int)
      for j in range(len(interval_points)-1):
        indices = range(interval_points[j], interval_points[j+1])
        X_batch = X[indices, :]
        y_batch = y[indices]
        w_batch = w[indices]
        ids_batch = ids[indices]
        (X_batch, y_batch, w_batch, ids_batch) = self._pad_batch(
            X_batch, y_batch, w_batch, ids_batch, batch_size)
        yield (X_batch, y_batch, w_batch, ids_batch)

  @staticmethod
  def from_numpy(data_dir, tasks, X, y, w, ids):
    raw_data = (ids, X, y, w)
    return Dataset(data_dir=data_dir, tasks=tasks, raw_data=raw_data)
    
  def to_numpy(self):
    """
    Transforms internal data into arrays X, y, w

    Creates three arrays containing all data in this object. This operation is
    dangerous (!) for large datasets which don't fit into memory.
    """
    Xs, ys, ws, ids = [], [], [], []
    for (X_b, y_b, w_b, ids_b) in self.itershards():
      Xs.append(X_b)
      ys.append(y_b)
      ws.append(w_b)
      ids.append(np.squeeze(ids_b))
    return (np.vstack(Xs), np.vstack(ys), np.vstack(ws),
            np.concatenate(ids))

  def get_labels(self):
    """
    Returns all labels for this dataset.
    """
    ys = []
    for (_, y_b, _, _) in self.itershards():
      ys.append(y_b)
    return np.vstack(ys)

  def get_weights(self):
    """
    Returns all weights for this dataset.
    """
    ws = []
    for (_, _, w_b, _) in self.itershards():
      ws.append(w_b)
    return np.vstack(ws)

  def _pad_batch(self, X_b, y_b, w_b, ids_b, batch_size):
    """Fix batch to have exactly batch_size elements.
 
    Due to rounding issues, some batches will not have exactly batch_size
    elements. Handle these batches by zero padding all arrays.
    """
    n, feature_shape = np.shape(X_b)[0], np.shape(X_b)[1:]
    _, num_tasks = np.shape(y_b)
    if n == batch_size:
      return (X_b, y_b, w_b, ids_b)
    else:
      X_batch = np.zeros((batch_size,) + feature_shape)
      y_batch = np.zeros((batch_size, num_tasks))
      w_batch = np.zeros((batch_size, num_tasks))
      ids_batch = np.zeros((batch_size,), dtype=object)
      X_batch[:n] = X_b
      y_batch[:n] = y_b
      w_batch[:n] = w_b
      ids_batch[:n] = ids_b
    return X_batch, y_batch, w_batch, ids_batch

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

  def get_statistics(self):
    """Computes and returns statistics of this dataset"""
    self.update_moments()
    df = self.metadata_df
    X_means, X_stds, y_means, y_stds = compute_mean_and_std(df)
    return X_means, X_stds, y_means, y_stds
  
  def update_moments(self):
    """Re-compute statistics of this dataset during transformation"""
    df = self.metadata_df
    update_mean_and_std(df)
 

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

def write_dataset_single(val, data_dir, feature_types=None, tasks=None,
                         raw_data=None, basename=None):
  """Writes files for single row (X, y, w, X-transformed, ...) to disk."""
  if feature_types is not None and tasks is not None:
    (df_file, df) = val
    # TODO(rbharath): This is a hack. clean up.
    if not len(df):
      return None
    ids, X, y, w = _df_to_numpy(df, feature_types, tasks)
  else:
    ids, X, y, w = raw_data
    df_file = ""
    assert X.shape[0] == y.shape[0]
    assert y.shape == w.shape
    assert len(ids) == X.shape[0]
  X_sums, X_sum_squares, X_n = compute_sums_and_nb_sample(X)
  y_sums, y_sum_squares, y_n = compute_sums_and_nb_sample(y, w)

  if feature_types is not None and tasks is not None:
    basename = os.path.splitext(os.path.basename(df_file))[0]
  out_X = os.path.join(data_dir, "%s-X.joblib" % basename)
  out_X_transformed = os.path.join(data_dir, "%s-X-transformed.joblib" % basename)
  out_X_sums = os.path.join(data_dir, "%s-X_sums.joblib" % basename)
  out_X_sum_squares = os.path.join(data_dir, "%s-X_sum_squares.joblib" % basename)
  out_X_n = os.path.join(data_dir, "%s-X_n.joblib" % basename)
  out_y = os.path.join(data_dir, "%s-y.joblib" % basename)
  out_y_transformed = os.path.join(data_dir, "%s-y-transformed.joblib" % basename)
  out_y_sums = os.path.join(data_dir, "%s-y_sums.joblib" % basename)
  out_y_sum_squares = os.path.join(data_dir, "%s-y_sum_squares.joblib" % basename)
  out_y_n = os.path.join(data_dir, "%s-y_n.joblib" % basename)
  out_w = os.path.join(data_dir, "%s-w.joblib" % basename)
  out_w_transformed = os.path.join(data_dir, "%s-w-transformed.joblib" % basename)
  out_ids = os.path.join(data_dir, "%s-ids.joblib" % basename)

  save_to_disk(X, out_X)
  save_to_disk(y, out_y)
  save_to_disk(w, out_w)
  # Write moments to disk
  save_to_disk(X_sums, out_X_sums)
  save_to_disk(X_sum_squares, out_X_sum_squares)
  save_to_disk(X_n, out_X_n)
  save_to_disk(y_sums, out_y_sums)
  save_to_disk(y_sum_squares, out_y_sum_squares)
  save_to_disk(y_n, out_y_n)
  # Write X, y as transformed versions
  save_to_disk(X, out_X_transformed)
  save_to_disk(y, out_y_transformed)
  save_to_disk(w, out_w_transformed)
  save_to_disk(ids, out_ids)
  return([df_file, tasks, out_ids, out_X, out_X_transformed, out_y,
          out_y_transformed, out_w, out_w_transformed,
          out_X_sums, out_X_sum_squares, out_X_n,
          out_y_sums, out_y_sum_squares, out_y_n])

# TODO(rbharath): This function is complicated enough that it should have unit
# tests.
def _df_to_numpy(df, feature_types, tasks):
  """Transforms a featurized dataset df into standard set of numpy arrays"""
  if not set(feature_types).issubset(df.keys()):
    raise ValueError(
        "Featurized data does not support requested feature_types.")
  # perform common train/test split across all tasks
  n_samples = df.shape[0]
  n_tasks = len(tasks)
  n_features = None
  y = df[tasks].values
  y = np.reshape(y, (n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  missing = np.zeros_like(y).astype(int)
  tensors = []
  for ind in range(n_samples):
    datapoint = df.iloc[ind]
    feature_list = []
    for feature_type in feature_types:
      feature_list.append(datapoint[feature_type])
    try:
      features = np.squeeze(np.concatenate(feature_list))
      for feature_ind, val in enumerate(features):
        if features[feature_ind] == "":
          features[feature_ind] = 0.
      features = features.astype(float)
      n_features = features.shape[0]
    except ValueError:
      missing[ind, :] = 1
      continue
    for task in range(n_tasks):
      if y[ind, task] == "":
        missing[ind, task] = 1
    tensors.append(features)
  x = np.stack(tensors)
  sorted_ids = df["mol_id"]

  # Set missing data to have weight zero
  # TODO(rbharath): There's a better way to do this with numpy indexing
  for ind in range(n_samples):
    for task in range(n_tasks):
      if missing[ind, task]:
        y[ind, task] = 0.
        w[ind, task] = 0.

  return sorted_ids, x.astype(float), y.astype(float), w.astype(float)

def compute_mean_and_std(df):
  """
  Compute means/stds of X/y from sums/sum_squares of tensors.
  """

  X_sums = []
  X_sum_squares = []
  X_n = []
  for _, row in df.iterrows():
    Xs = load_from_disk(row['X_sums'])
    Xss = load_from_disk(row['X_sum_squares'])
    Xn = load_from_disk(row['X_n'])
    X_sums.append(np.array(Xs))
    X_sum_squares.append(np.array(Xss))
    X_n.append(np.array(Xn))

  # Note that X_n is a list of floats
  n = float(np.sum(X_n))
  X_sums = np.vstack(X_sums)
  X_sum_squares = np.vstack(X_sum_squares)
  overall_X_sums = np.sum(X_sums, axis=0)
  overall_X_means = overall_X_sums / n
  overall_X_sum_squares = np.sum(X_sum_squares, axis=0)

  X_vars = (overall_X_sum_squares - np.square(overall_X_sums)/n)/(n)

  y_sums = []
  y_sum_squares = []
  y_n = []
  for _, row in df.iterrows():
    ys = load_from_disk(row['y_sums'])
    yss = load_from_disk(row['y_sum_squares'])
    yn = load_from_disk(row['y_n'])
    y_sums.append(np.array(ys))
    y_sum_squares.append(np.array(yss))
    y_n.append(np.array(yn))

  # Note y_n is a list of arrays of shape (n_tasks,)
  y_n = np.sum(y_n, axis=0)
  y_sums = np.vstack(y_sums)
  y_sum_squares = np.vstack(y_sum_squares)
  y_means = np.sum(y_sums, axis=0)/y_n
  y_vars = np.sum(y_sum_squares, axis=0)/y_n - np.square(y_means)
  return overall_X_means, np.sqrt(X_vars), y_means, np.sqrt(y_vars)

def update_mean_and_std(df):
  """
  Compute means/stds of X/y from sums/sum_squares of tensors.
  """
  X_transform = []
  for _, row in df.iterrows():
    Xt = load_from_disk(row['X-transformed'])
    Xs = np.sum(Xt,axis=0)
    Xss = np.sum(np.square(Xt),axis=0)
    save_to_disk(Xs, row['X_sums'])
    save_to_disk(Xss, row['X_sum_squares'])

  y_transform = []
  for _, row in df.iterrows():
    yt = load_from_disk(row['y-transformed'])
    ys = np.sum(yt,axis=0)
    yss = np.sum(np.square(yt),axis=0)
    save_to_disk(ys, row['y_sums'])
    save_to_disk(yss, row['y_sum_squares'])

