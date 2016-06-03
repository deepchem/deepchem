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
from deepchem.utils.save import log

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

# TODO(rbharath): The semantics of this class are very difficult to debug.
# Multiple transformations of the data are performed on disk, and computations
# of mean/std are spread across multiple functions for efficiency. Some
# refactoring needs to happen here.
class Dataset(object):
  """
  Wrapper class for dataset transformed into X, y, w numpy ndarrays.
  """
  def __init__(self, data_dir=None, tasks=[], metadata_rows=None, #featurizers=None, 
               raw_data=None, verbosity=None, reload=False):
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    self.data_dir = data_dir
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity

    if not reload or not os.path.exists(self._get_metadata_filename()):
      log("About to start initializing dataset", self.verbosity)

      if metadata_rows is not None:
        self.metadata_df = Dataset.construct_metadata(metadata_rows)
        self.save_to_disk()
      elif raw_data is not None:
        metadata_rows = []
        ids, X, y, w = raw_data
        metadata_rows.append(
            Dataset.write_data_to_disk(self.data_dir, "data", tasks, X, y, w, ids))
        self.metadata_df = Dataset.construct_metadata(metadata_rows)
        self.save_to_disk()
      else:
        # Create an empty metadata dataframe to be filled at a later time
        basename = "metadata"
        metadata_rows = [Dataset.write_data_to_disk(
            self.data_dir, basename, tasks)]
        self.metadata_df = Dataset.construct_metadata(metadata_rows)
        self.save_to_disk()

    else:
      log("Loading pre-existing metadata file.", self.verbosity)
      if os.path.exists(self._get_metadata_filename()):
        self.metadata_df = load_from_disk(self._get_metadata_filename())
      else:
        raise ValueError("No metadata found.")

  @staticmethod
  def write_dataframe(val, data_dir, featurizers=None, tasks=None,
                      raw_data=None, basename=None):
    """Writes data from dataframe to disk."""
    if featurizers is not None and tasks is not None:
      feature_types = [featurizer.__class__.__name__ for featurizer in featurizers]
      (basename, df) = val
      # TODO(rbharath): This is a hack. clean up.
      if not len(df):
        return None
      ids, X, y, w = _df_to_numpy(df, feature_types, tasks)
    else:
      ids, X, y, w = raw_data
      basename = ""
      assert X.shape[0] == y.shape[0]
      assert y.shape == w.shape
      assert len(ids) == X.shape[0]
    return Dataset.write_data_to_disk(data_dir, basename, tasks, X, y, w, ids)

  @staticmethod
  def construct_metadata(metadata_entries):
    """Construct a dataframe containing metadata.
  
    metadata_entries should have elements returned by write_data_to_disk
    above.
    """
    metadata_df = pd.DataFrame(
        metadata_entries,
        columns=('basename','task_names', 'ids',
                 'X', 'X-transformed', 'y', 'y-transformed',
                 'w', 'w-transformed',
                 'X_sums', 'X_sum_squares', 'X_n',
                 'y_sums', 'y_sum_squares', 'y_n'))
    return metadata_df

  @staticmethod
  def write_data_to_disk(data_dir, basename, tasks, X=None, y=None, w=None, ids=None):
    out_X = "%s-X.joblib" % basename
    out_X_transformed = "%s-X-transformed.joblib" % basename
    out_X_sums = "%s-X_sums.joblib" % basename
    out_X_sum_squares = "%s-X_sum_squares.joblib" % basename
    out_X_n = "%s-X_n.joblib" % basename
    out_y = "%s-y.joblib" % basename
    out_y_transformed = "%s-y-transformed.joblib" % basename
    out_y_sums = "%s-y_sums.joblib" % basename
    out_y_sum_squares = "%s-y_sum_squares.joblib" % basename
    out_y_n = "%s-y_n.joblib" % basename
    out_w = "%s-w.joblib" % basename
    out_w_transformed = "%s-w-transformed.joblib" % basename
    out_ids = "%s-ids.joblib" % basename

    if X is not None:
      ############################################## DEBUG
      print("X.shape")
      print(X.shape)
      print("os.path.join(data_dir, out_X)")
      print(os.path.join(data_dir, out_X))
      ############################################## DEBUG
      save_to_disk(X, os.path.join(data_dir, out_X))
      save_to_disk(X, os.path.join(data_dir, out_X_transformed))
      X_sums, X_sum_squares, X_n = compute_sums_and_nb_sample(X)
      save_to_disk(X_sums, os.path.join(data_dir, out_X_sums))
      save_to_disk(X_sum_squares, os.path.join(data_dir, out_X_sum_squares))
      save_to_disk(X_n, os.path.join(data_dir, out_X_n))
    if y is not None:
      save_to_disk(y, os.path.join(data_dir, out_y))
      save_to_disk(y, os.path.join(data_dir, out_y_transformed))
      y_sums, y_sum_squares, y_n = compute_sums_and_nb_sample(y, w)
      save_to_disk(y_sums, os.path.join(data_dir, out_y_sums))
      save_to_disk(y_sum_squares, os.path.join(data_dir, out_y_sum_squares))
      save_to_disk(y_n, os.path.join(data_dir, out_y_n))
    if w is not None:
      save_to_disk(w, os.path.join(data_dir, out_w))
      save_to_disk(w, os.path.join(data_dir, out_w_transformed))
    if ids is not None:
      save_to_disk(ids, os.path.join(data_dir, out_ids))
    return [basename, tasks, out_ids, out_X, out_X_transformed, out_y,
            out_y_transformed, out_w, out_w_transformed,
            out_X_sums, out_X_sum_squares, out_X_n,
            out_y_sums, out_y_sum_squares, out_y_n]
  

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
        os.path.join(
            self.data_dir,
            self.metadata_df.iterrows().next()[1]['X-transformed']))[0]
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
      X = np.array(load_from_disk(
          os.path.join(self.data_dir, row['X-transformed'])))
      y = np.array(load_from_disk(
          os.path.join(self.data_dir, row['y-transformed'])))
      w = np.array(load_from_disk(
          os.path.join(self.data_dir, row['w-transformed'])))
      ids = np.array(load_from_disk(
          os.path.join(self.data_dir, row['ids'])), dtype=object)
      yield (X, y, w, ids)

  def iterbatches(self, batch_size=None, epoch=0):
    """
    Returns minibatches from dataset.
    """
    for i, (X, y, w, ids) in enumerate(self.itershards()):
      nb_sample = np.shape(X)[0]
      if batch_size is None:
        shard_batch_size = nb_sample
      else:
        shard_batch_size = batch_size 
      interval_points = np.linspace(
          0, nb_sample, np.ceil(float(nb_sample)/shard_batch_size)+1, dtype=int)
      for j in range(len(interval_points)-1):
        indices = range(interval_points[j], interval_points[j+1])
        X_batch = X[indices, :]
        y_batch = y[indices]
        w_batch = w[indices]
        ids_batch = ids[indices]
        yield (X_batch, y_batch, w_batch, ids_batch)

  @staticmethod
  def from_numpy(data_dir, X, y, w=None, ids=None, tasks=None):
    n_samples = len(X)
    # The -1 indicates that y will be reshaped to have length -1
    if n_samples > 0:
      y = np.reshape(y, (n_samples, -1))
      w = np.reshape(w, (n_samples, -1))
    n_tasks = y.shape[1]
    if ids is None:
      ids = np.arange(n_samples)
    if w is None:
      w = np.ones_like(y)
    if tasks is None:
      tasks = np.arange(n_tasks)
    raw_data = (ids, X, y, w)
    return Dataset(data_dir=data_dir, tasks=tasks, raw_data=raw_data)

  def select(self, select_dir, indices):
    """Creates a new dataset from a selection of indices from self."""
    indices = np.array(indices).astype(int)
    X, y, w, ids = self.to_numpy()
    tasks = self.get_task_names()
    X_sel, y_sel, w_sel, ids_sel = (
        X[indices], y[indices], w[indices], ids[indices])
    return Dataset.from_numpy(select_dir, X_sel, y_sel, w_sel, ids_sel, tasks)
    
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
      ids.append(np.atleast_1d(np.squeeze(ids_b)))
    np.concatenate(ids)
    return (np.vstack(Xs), np.vstack(ys), np.vstack(ws),
            np.concatenate(ids))

  def get_ids(self):
    """
    Returns all molecule-ids for this dataset.
    """
    ids = []
    for (_, _, _, ids_b) in self.itershards():
      ids.append(np.atleast_1d(np.squeeze(ids_b)))
    return np.concatenate(ids)

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
      ws.append(np.array(w_b))
    return np.vstack(ws)

  def __len__(self):
    """
    Finds number of elements in dataset.
    """
    total = 0
    for _, row in self.metadata_df.iterrows():
      y = load_from_disk(os.path.join(self.data_dir, row['y-transformed']))
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
    if len(self) == 0:
      return None, None, None, None
    self.update_moments()
    df = self.metadata_df
    X_means, X_stds, y_means, y_stds = self._compute_mean_and_std(df)
    return X_means, X_stds, y_means, y_stds

  def _compute_mean_and_std(self, df):
    """
    Compute means/stds of X/y from sums/sum_squares of tensors.
    """

    X_sums = []
    X_sum_squares = []
    X_n = []
    for _, row in df.iterrows():
      Xs = load_from_disk(os.path.join(self.data_dir, row['X_sums']))
      Xss = load_from_disk(os.path.join(self.data_dir, row['X_sum_squares']))
      Xn = load_from_disk(os.path.join(self.data_dir, row['X_n']))
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
      ys = load_from_disk(os.path.join(self.data_dir, row['y_sums']))
      yss = load_from_disk(os.path.join(self.data_dir, row['y_sum_squares']))
      yn = load_from_disk(os.path.join(self.data_dir, row['y_n']))
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

  
  def update_moments(self):
    """Re-compute statistics of this dataset during transformation"""
    df = self.metadata_df
    self._update_mean_and_std(df)

  def _update_mean_and_std(self, df):
    """
    Compute means/stds of X/y from sums/sum_squares of tensors.
    """
    X_transform = []
    for _, row in df.iterrows():
      Xt = load_from_disk(os.path.join(self.data_dir, row['X-transformed']))
      Xs = np.sum(Xt,axis=0)
      Xss = np.sum(np.square(Xt),axis=0)
      save_to_disk(Xs, os.path.join(self.data_dir, row['X_sums']))
      save_to_disk(Xss, os.path.join(self.data_dir, row['X_sum_squares']))

    y_transform = []
    for _, row in df.iterrows():
      yt = load_from_disk(os.path.join(self.data_dir, row['y-transformed']))
      ys = np.sum(yt,axis=0)
      yss = np.sum(np.square(yt),axis=0)
      save_to_disk(ys, os.path.join(self.data_dir, row['y_sums']))
      save_to_disk(yss, os.path.join(self.data_dir, row['y_sum_squares']))

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
def _df_to_numpy(df, feature_types, tasks):
  """Transforms a featurized dataset df into standard set of numpy arrays"""
  if not set(feature_types).issubset(df.keys()):
    raise ValueError(
        "Featurized data does not support requested feature_types.")
  # perform common train/test split across all tasks
  n_samples = df.shape[0]
  n_tasks = len(tasks)
  y = np.hstack([
      np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])
  w = np.ones((n_samples, n_tasks))
  missing = np.zeros_like(y).astype(int)
  tensors = []
  feature_shape = None
  for ind in range(n_samples):
    datapoint = df.iloc[ind]
    feature_list = []
    for feature_type in feature_types:
      feature_list.append(datapoint[feature_type])
    try:
      features = np.squeeze(np.concatenate(feature_list))
      if features.size == 0:
        features = np.zeros(feature_shape)
        tensors.append(features)
        missing[ind, :] = 1
        continue
      for feature_ind, val in enumerate(features):
        if features[feature_ind] == "":
          features[feature_ind] = 0.
      features = features.astype(float)
      if feature_shape is None:
        feature_shape = features.shape
    except ValueError:
      missing[ind, :] = 1
      continue
    for task in range(n_tasks):
      if y[ind, task] == "":
        missing[ind, task] = 1
    if features.shape != feature_shape:
      missing[ind, :] = 1
      continue
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

  # Adding this assertion in to avoid ill-formed outputs.
  assert len(sorted_ids) == len(x) == len(y) == len(w)
  return sorted_ids, x.astype(float), y.astype(float), w.astype(float)
