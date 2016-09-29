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
import random
from functools import partial
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import log
import tempfile
import time
import shutil

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

def sparsify_features(X):
  """Extracts a sparse feature representation from dense feature array."""
  n_samples = len(X)
  X_sparse = []
  for i in range(n_samples):
    nonzero_inds = np.nonzero(X[i])[0]
    nonzero_vals = X[i][nonzero_inds]
    X_sparse.append((nonzero_inds, nonzero_vals))
  X_sparse = np.array(X_sparse, dtype=object)
  return X_sparse

def densify_features(X_sparse, num_features):
  """Expands sparse feature representation to dense feature array."""
  n_samples = len(X_sparse)
  X = np.zeros((n_samples, num_features))
  for i in range(n_samples):
    nonzero_inds, nonzero_vals = X_sparse[i]
    X[i][nonzero_inds.astype(int)] = nonzero_vals
  return X

def pad_features(batch_size, X_b):
  """Pads a batch of features to have precisely batch_size elements.
  
  Version of pad_batch for use at prediction time.
  """
  num_samples = len(X_b)
  if num_samples == batch_size:
    return X_b
  else:
    # By invariant of when this is called, can assume num_samples > 0
    # and num_samples < batch_size
    if len(X_b.shape) > 1:
      feature_shape = X_b.shape[1:]
      X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
    else:
      X_out = np.zeros((batch_size,), dtype=X_b.dtype)

    # Fill in batch arrays 
    start = 0 
    while start < batch_size:
      num_left = batch_size - start 
      if num_left < num_samples:
        increment = num_left
      else:
        increment = num_samples
      X_out[start:start+increment] = X_b[:increment]
      start += increment
    return X_out

def pad_batch(batch_size, X_b, y_b, w_b, ids_b):
  """Pads batch to have size precisely batch_size elements.

  Fills in batch by wrapping around samples till whole batch is filled.
  """
  num_samples = len(X_b)
  if num_samples == batch_size:
    return (X_b, y_b, w_b, ids_b)
  else:
    # By invariant of when this is called, can assume num_samples > 0
    # and num_samples < batch_size
    if len(X_b.shape) > 1:
      feature_shape = X_b.shape[1:]
      X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
    else:
      X_out = np.zeros((batch_size,), dtype=X_b.dtype)

    num_tasks = y_b.shape[1]
    y_out = np.zeros((batch_size, num_tasks), dtype=y_b.dtype) 
    w_out = np.zeros((batch_size, num_tasks), dtype=w_b.dtype)
    ids_out = np.zeros((batch_size,), dtype=ids_b.dtype)

    # Fill in batch arrays 
    start = 0 
    while start < batch_size:
      num_left = batch_size - start 
      if num_left < num_samples:
        increment = num_left
      else:
        increment = num_samples
      X_out[start:start+increment] = X_b[:increment]
      y_out[start:start+increment] = y_b[:increment]
      w_out[start:start+increment] = w_b[:increment]
      ids_out[start:start+increment] = ids_b[:increment]
      start += increment
    return (X_out, y_out, w_out, ids_out)

class Dataset(object):
  """
  Wrapper class for dataset transformed into X, y, w numpy ndarrays.
  """
  def __init__(self, data_dir=None, tasks=[], metadata_rows=None, #featurizers=None, 
               raw_data=None, verbosity=None, reload=False,
               compute_feature_statistics=True):
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    self.data_dir = data_dir
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity

    if not reload or not os.path.exists(self._get_metadata_filename()):
      if metadata_rows is not None:
        self.metadata_df = Dataset.construct_metadata(metadata_rows)
        self.save_to_disk()
      elif raw_data is not None:
        metadata_rows = []
        ids, X, y, w = raw_data
        metadata_rows.append(
            Dataset.write_data_to_disk(
                self.data_dir, "data", tasks, X, y, w, ids,
                compute_feature_statistics=compute_feature_statistics))
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
  def write_dataframe(val, data_dir, featurizer=None, tasks=None,
                      raw_data=None, basename=None, mol_id_field="mol_id",
                      verbosity=None, compute_feature_statistics=None):
    """Writes data from dataframe to disk."""
    if featurizer is not None and tasks is not None:
      feature_type = featurizer.__class__.__name__
      (basename, df) = val
      # TODO(rbharath): This is a hack. clean up.
      if not len(df):
        return None
      if compute_feature_statistics is None:
        if hasattr(featurizer, "dtype"):
          dtype = featurizer.dtype
          compute_feature_statistics = False
        else:
          dtype = float
          compute_feature_statistics = True
      ############################################################## TIMING
      time1 = time.time()
      ############################################################## TIMING
      ids, X, y, w = convert_df_to_numpy(df, feature_type, tasks, mol_id_field,
                                         dtype, verbosity)
      ############################################################## TIMING
      time2 = time.time()
      log("TIMING: convert_df_to_numpy took %0.3f s" % (time2-time1), verbosity)
      ############################################################## TIMING
    else:
      ids, X, y, w = raw_data
      basename = ""
      assert X.shape[0] == y.shape[0]
      assert y.shape == w.shape
      assert len(ids) == X.shape[0]
    return Dataset.write_data_to_disk(
        data_dir, basename, tasks, X, y, w, ids,
        compute_feature_statistics=compute_feature_statistics)

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
  def write_data_to_disk(data_dir, basename, tasks, X=None, y=None, w=None, ids=None,
                         compute_feature_statistics=True):
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
      save_to_disk(X, os.path.join(data_dir, out_X))
      save_to_disk(X, os.path.join(data_dir, out_X_transformed))
      if compute_feature_statistics:
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
    return next(self.metadata_df.iterrows())[1]['task_names']

  def get_data_shape(self):
    """
    Gets array shape of datapoints in this dataset.
    """
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_X = load_from_disk(
        os.path.join(
            self.data_dir,
            next(self.metadata_df.iterrows())[1]['X-transformed']))[0]
    return np.shape(sample_X)

  def get_shard_size(self):
    """Gets size of shards on disk."""
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_y = load_from_disk(
        os.path.join(
            self.data_dir,
            next(self.metadata_df.iterrows())[1]['y-transformed']))
    return len(sample_y)

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

  def iterbatches(self, batch_size=None, epoch=0, deterministic=False,
                  pad_batches=False):
    """Returns minibatches from dataset randomly."""
    num_shards = self.get_number_shards()
    if not deterministic:
      shard_perm = np.random.permutation(num_shards)
    else:
      shard_perm = np.arange(num_shards)
    for i in range(num_shards):
      X, y, w, ids = self.get_shard(shard_perm[i])
      n_samples = X.shape[0]
      if not deterministic:
        sample_perm = np.random.permutation(n_samples)
      else:
        sample_perm = np.arange(n_samples)
      if batch_size is None:
        shard_batch_size = n_samples
      else:
        shard_batch_size = batch_size 
      interval_points = np.linspace(
          0, n_samples, np.ceil(float(n_samples)/shard_batch_size)+1, dtype=int)
      for j in range(len(interval_points)-1):
        indices = range(interval_points[j], interval_points[j+1])
        perm_indices = sample_perm[indices]
        X_batch = X[perm_indices, :]
        y_batch = y[perm_indices]
        w_batch = w[perm_indices]
        ids_batch = ids[perm_indices]
        if pad_batches:
          (X_batch, y_batch, w_batch, ids_batch) = pad_batch(
            shard_batch_size, X_batch, y_batch, w_batch, ids_batch)
        yield (X_batch, y_batch, w_batch, ids_batch)

  def reshard(self, shard_size):
    """Reshards data to have specified shard size."""
    # Create temp directory to store resharded version
    reshard_dir = tempfile.mkdtemp()
    new_metadata = []
    # Write data in new shards
    ind = 0
    tasks = self.get_task_names() 
    X_next = np.zeros((0,) + self.get_data_shape())
    y_next = np.zeros((0,) + (len(tasks),))
    w_next = np.zeros((0,) + (len(tasks),))
    ids_next = np.zeros((0,), dtype=object)
    for (X, y, w, ids) in self.itershards():
      X_next = np.vstack([X_next, X])
      y_next = np.vstack([y_next, y])
      w_next = np.vstack([w_next, w])
      ids_next = np.concatenate([ids_next, ids])
      while len(X_next) > shard_size:
        X_batch, X_next = X_next[:shard_size], X_next[shard_size:]
        y_batch, y_next = y_next[:shard_size], y_next[shard_size:]
        w_batch, w_next = w_next[:shard_size], w_next[shard_size:]
        ids_batch, ids_next = ids_next[:shard_size], ids_next[shard_size:]
        new_basename = "reshard-%d" % ind
        new_metadata.append(Dataset.write_data_to_disk(
            reshard_dir, new_basename, tasks, X_batch, y_batch, w_batch, ids_batch))
        ind += 1
    # Handle spillover from last shard
    new_basename = "reshard-%d" % ind
    new_metadata.append(Dataset.write_data_to_disk(
        reshard_dir, new_basename, tasks, X_next, y_next, w_next, ids_next))
    ind += 1
    # Get new metadata rows
    resharded_dataset = Dataset(
        data_dir=reshard_dir, tasks=tasks, metadata_rows=new_metadata,
        verbosity=self.verbosity)
    shutil.rmtree(self.data_dir)
    shutil.move(reshard_dir, self.data_dir)
    self.metadata_df = resharded_dataset.metadata_df
    self.save_to_disk()

  @staticmethod
  def from_numpy(data_dir, X, y, w=None, ids=None, tasks=None, verbosity=None,
                 compute_feature_statistics=True):
    n_samples = len(X)
    # The -1 indicates that y will be reshaped to have length -1
    if n_samples > 0:
      y = np.reshape(y, (n_samples, -1))
      if w is not None:
        w = np.reshape(w, (n_samples, -1))
    n_tasks = y.shape[1]
    if ids is None:
      ids = np.arange(n_samples)
    if w is None:
      w = np.ones_like(y)
    if tasks is None:
      tasks = np.arange(n_tasks)
    raw_data = (ids, X, y, w)
    return Dataset(data_dir=data_dir, tasks=tasks, raw_data=raw_data,
                   verbosity=verbosity,
                   compute_feature_statistics=compute_feature_statistics)

  @staticmethod
  def merge(merge_dir, datasets):
    """Merges provided datasets into a merged dataset."""
    if not os.path.exists(merge_dir):
      os.makedirs(merge_dir)
    Xs, ys, ws, all_ids = [], [], [], []
    metadata_rows = []
    for ind, dataset in enumerate(datasets):
      X, y, w, ids = dataset.to_numpy()
      basename = "dataset-%d" % ind
      tasks = dataset.get_task_names()
      metadata_rows.append(
          Dataset.write_data_to_disk(merge_dir, basename, tasks, X, y, w, ids))
    return Dataset(data_dir=merge_dir,
                   metadata_rows=metadata_rows,
                   verbosity=dataset.verbosity)

  def subset(self, subset_dir, shard_nums):
    """Creates a subset of the original dataset on disk."""
    if not os.path.exists(subset_dir):
      os.makedirs(subset_dir)
    tasks = self.get_task_names()
    metadata_rows = []
    for shard_num, row in self.metadata_df.iterrows():
      if shard_num not in shard_nums:
        continue
      X, y, w, ids = self.get_shard(shard_num)
      basename = "dataset-%d" % shard_num
      metadata_rows.append(Dataset.write_data_to_disk(
          subset_dir, basename, tasks, X, y, w, ids))
    return Dataset(data_dir=subset_dir,
                   metadata_rows=metadata_rows,
                   verbosity=self.verbosity)

  def reshard_shuffle(self, reshard_size=10, num_reshards=3):
    """Shuffles by resharding, shuffling shards, undoing resharding."""
    #########################################################  TIMING
    time1 = time.time()
    #########################################################  TIMING
    for i in range(num_reshards):
      orig_shard_size = self.get_shard_size()
      log("Resharding to shard-size %d." % reshard_size, self.verbosity)
      self.reshard(shard_size=reshard_size)
      log("Shuffling shard order.", self.verbosity)
      self.shuffle_shards()
      log("Resharding to original shard-size %d." % orig_shard_size,
          self.verbosity)
      self.reshard(shard_size=orig_shard_size)
      self.shuffle_each_shard()
    #########################################################  TIMING
    time2 = time.time()
    log("TIMING: reshard_shuffle took %0.3f s" % (time2-time1),
        self.verbosity)
    #########################################################  TIMING

  def sparse_shuffle(self):
    """Shuffling that exploits data sparsity to shuffle large datasets.

    Only for 1-dimensional feature vectors (does not work for tensorial
    featurizations).
    """
    #########################################################  TIMING
    time1 = time.time()
    #########################################################  TIMING
    shard_size = self.get_shard_size()
    num_shards = self.get_number_shards()
    X_sparses, ys, ws, ids = [], [], [], []
    num_features = None
    for i in range(num_shards):
      (X_s, y_s, w_s, ids_s) = self.get_shard(i) 
      if num_features is None:
        num_features = X_s.shape[1]
      X_sparse = sparsify_features(X_s) 
      X_sparses, ys, ws, ids = (
          X_sparses + [X_sparse], ys + [y_s], ws + [w_s],
          ids + [np.atleast_1d(np.squeeze(ids_s))])
    # Get full dataset in memory
    (X_sparse, y, w, ids) = (
        np.vstack(X_sparses), np.vstack(ys), np.vstack(ws), np.concatenate(ids))
    # Shuffle in memory
    num_samples = len(X_sparse)
    permutation = np.random.permutation(num_samples)
    X_sparse, y, w, ids = (X_sparse[permutation], y[permutation],
                           w[permutation], ids[permutation])
    # Write shuffled shards out to disk
    for i in range(num_shards):
      start, stop = i*shard_size, (i+1)*shard_size
      (X_sparse_s, y_s, w_s, ids_s) = (
          X_sparse[start:stop], y[start:stop], w[start:stop], ids[start:stop])
      X_s = densify_features(X_sparse_s, num_features)
      self.set_shard(i, X_s, y_s, w_s, ids_s)
    #########################################################  TIMING
    time2 = time.time()
    log("TIMING: sparse_shuffle took %0.3f s" % (time2-time1),
        self.verbosity)
    #########################################################  TIMING

  def shuffle(self, iterations=1):
    """Shuffles this dataset on disk to have random order."""
    #np.random.seed(9452)
    for _ in range(iterations):
      metadata_rows = []
      tasks = self.get_task_names()
      # Shuffle the arrays corresponding to each row in metadata_df
      n_rows = len(self.metadata_df.index)
      len_data = len(self)
      print("ABOUT TO SHUFFLE DATA ONCE")
      for i in range(n_rows):
        # Select random row to swap with
        j = np.random.randint(n_rows)
        row_i, row_j = self.metadata_df.iloc[i], self.metadata_df.iloc[j]
        metadata_rows.append(row_i)
        # Useful to avoid edge cases, but perhaps there's a better solution
        if i == j:
          continue
        basename_i, basename_j = row_i["basename"], row_j["basename"]
        X_i, y_i, w_i, ids_i = self.get_shard(i)
        X_j, y_j, w_j, ids_j = self.get_shard(j)
        n_i, n_j = X_i.shape[0], X_j.shape[0]

        # Join two shards and shuffle them at random.
        X = np.vstack([X_i, X_j])
        y = np.vstack([y_i, y_j])
        w = np.vstack([w_i, w_j])
        ids = np.concatenate([ids_i, ids_j])
        permutation = np.random.permutation(n_i + n_j)
        X, y, w, ids = (X[permutation], y[permutation],
                        w[permutation], ids[permutation])

        X_i, y_i, w_i, ids_i = X[:n_i], y[:n_i], w[:n_i], ids[:n_i]
        X_j, y_j, w_j, ids_j = X[n_i:], y[n_i:], w[n_i:], ids[n_i:]

        Dataset.write_data_to_disk(
            self.data_dir, basename_i, tasks, X_i, y_i, w_i, ids_i)
        Dataset.write_data_to_disk(
            self.data_dir, basename_j, tasks, X_j, y_j, w_j, ids_j)
        assert len(self) == len_data
      # Now shuffle order of rows in metadata_df
      random.shuffle(metadata_rows)
      self.metadata_df = Dataset.construct_metadata(metadata_rows)
      self.save_to_disk()

  def shuffle_each_shard(self):
    """Shuffles elements within each shard of the datset."""
    tasks = self.get_task_names()
    # Shuffle the arrays corresponding to each row in metadata_df
    n_rows = len(self.metadata_df.index)
    n_rows = len(self.metadata_df.index)
    for i in range(n_rows):
      row = self.metadata_df.iloc[i]
      basename = row["basename"]
      X, y, w, ids = self.get_shard(i)
      n = X.shape[0]
      permutation = np.random.permutation(n)
      X, y, w, ids = (X[permutation], y[permutation],
                      w[permutation], ids[permutation])
      Dataset.write_data_to_disk(
          self.data_dir, basename, tasks, X, y, w, ids)

  def shuffle_shards(self):
    """Shuffles the order of the shards for this dataset."""
    metadata_rows = self.metadata_df.values.tolist()
    random.shuffle(metadata_rows)
    self.metadata_df = Dataset.construct_metadata(metadata_rows)
    self.save_to_disk()

  def get_shard(self, i):
    """Retrieves data for the i-th shard from disk."""
    row = self.metadata_df.iloc[i]
    X = np.array(load_from_disk(
        os.path.join(self.data_dir, row['X-transformed'])))
    y = np.array(load_from_disk(
        os.path.join(self.data_dir, row['y-transformed'])))
    w = np.array(load_from_disk(
        os.path.join(self.data_dir, row['w-transformed'])))
    ids = np.array(load_from_disk(
        os.path.join(self.data_dir, row['ids'])), dtype=object)
    return (X, y, w, ids)

  def set_shard(self, shard_num, X, y, w, ids):
    """Writes data shard to disk"""
    basename = "shard-%d" % shard_num 
    tasks = self.get_task_names()
    Dataset.write_data_to_disk(self.data_dir, basename, tasks, X, y, w, ids)

  def set_verbosity(self, new_verbosity):
    """Sets verbosity."""
    self.verbosity = new_verbosity

  # TODO(rbharath): This change for general object types seems a little
  # kludgey.  Is there a more principled approach to support general objects?
  def select(self, select_dir, indices, compute_feature_statistics=False):
    """Creates a new dataset from a selection of indices from self.

    Parameters
    ----------
    select_dir: string
      Path to new directory that the selected indices will be copied to.
    indices: list
      List of indices to select.
    compute_feature_statistics: bool
      Whether or not to compute moments of features. Only meaningful if features
      are np.ndarrays. Not meaningful for other featurizations.
    """
    if not os.path.exists(select_dir):
      os.makedirs(select_dir)
    # Handle edge case with empty indices
    if not len(indices):
      return Dataset(
          data_dir=select_dir, metadata_rows=[], verbosity=self.verbosity)
    indices = np.array(sorted(indices)).astype(int)
    count, indices_count = 0, 0
    metadata_rows = []
    tasks = self.get_task_names()
    for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
      shard_len = len(X)
      # Find indices which rest in this shard
      num_shard_elts = 0
      while indices[indices_count+num_shard_elts] < count + shard_len:
        num_shard_elts += 1
        if indices_count + num_shard_elts >= len(indices):
          break
      # Need to offset indices to fit within shard_size
      shard_inds =  indices[indices_count:indices_count+num_shard_elts] - count
      X_sel = X[shard_inds]
      y_sel = y[shard_inds]
      w_sel = w[shard_inds]
      ids_sel = ids[shard_inds]
      basename = "dataset-%d" % shard_num
      metadata_rows.append(
          Dataset.write_data_to_disk(
              select_dir, basename, tasks,
              X_sel, y_sel, w_sel, ids_sel,
              compute_feature_statistics=compute_feature_statistics))
      # Updating counts
      indices_count += num_shard_elts
      count += shard_len
    return Dataset(data_dir=select_dir,
                   metadata_rows=metadata_rows,
                   verbosity=self.verbosity)

  def to_singletask(self, task_dirs):
    """Transforms multitask dataset in collection of singletask datasets."""
    tasks = self.get_task_names()
    assert len(tasks) == len(task_dirs)
    log("Splitting multitask dataset into singletask datasets", self.verbosity)
    task_metadata_rows = {task: [] for task in tasks}
    for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
      log("Processing shard %d" % shard_num, self.verbosity)
      basename = "dataset-%d" % shard_num
      for task_num, task in enumerate(tasks):
        log("\tTask %s" % task, self.verbosity)
        w_task = w[:, task_num]
        y_task = y[:, task_num]

        # Extract those datapoints which are present for this task
        X_nonzero = X[w_task != 0]
        num_datapoints = X_nonzero.shape[0]
        y_nonzero = np.reshape(y_task[w_task != 0], (num_datapoints, 1))
        w_nonzero = np.reshape(w_task[w_task != 0], (num_datapoints, 1))
        ids_nonzero = ids[w_task != 0]

        if X_nonzero.size > 0: 
          task_metadata_rows[task].append(
            Dataset.write_data_to_disk(
                task_dirs[task_num], basename, [task],
                X_nonzero, y_nonzero, w_nonzero, ids_nonzero))
    
    task_datasets = [
        Dataset(data_dir=task_dirs[task_num],
                metadata_rows=task_metadata_rows[task],
                verbosity=self.verbosity)
        for (task_num, task) in enumerate(tasks)]
    return task_datasets
    
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
    if len(self) == 0:
      return np.array([])
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

  def get_shape(self):
    """Finds shape of dataset."""
    n_tasks = len(self.get_task_names())
    X_shape = np.array((0,) + (0,) * len(self.get_data_shape())) 
    y_shape = np.array((0,) + (0,))
    w_shape = np.array((0,) + (0,))
    ids_shape = np.array((0,))
    for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
      if shard_num == 0:
        X_shape += np.array(X.shape)
        y_shape += np.array(y.shape)
        w_shape += np.array(w.shape)
        ids_shape += np.array(ids.shape)
      else:
        X_shape[0] += np.array(X.shape)[0]
        y_shape[0] += np.array(y.shape)[0]
        w_shape[0] += np.array(w.shape)[0]
        ids_shape[0] += np.array(ids.shape)[0]
    return tuple(X_shape), tuple(y_shape), tuple(w_shape), tuple(ids_shape)

  def get_label_means(self):
    """Return pandas series of label means."""
    return self.metadata_df["y_means"]

  def get_label_stds(self):
    """Return pandas series of label stds."""
    return self.metadata_df["y_stds"]

  def get_statistics(self, X_stats=True, y_stats=True):
    """Computes and returns statistics of this dataset"""
    if len(self) == 0:
      return None, None, None, None
    self.update_moments(X_stats, y_stats)
    df = self.metadata_df
    if X_stats and not y_stats:
      X_means, X_stds = self._compute_mean_and_std(df, X_stats, y_stats)
      return X_means, X_stds
    elif y_stats and not X_stats:
      y_means, y_stds = self._compute_mean_and_std(df, X_stats, y_stats)
      return y_means, y_stds
    elif X_stats and y_stats:
      X_means, X_stds = self._compute_mean_and_std(
          df, X_stats=True, y_stats=False)
      y_means, y_stds = self._compute_mean_and_std(
          df, X_stats=False, y_stats=True)
      return X_means, X_stds, y_means, y_stds
    else:
      return None

  def _compute_mean_and_std(self, df, X_stats, y_stats):
    """
    Compute means/stds of X/y from sums/sum_squares of tensors.
    """

    if X_stats:
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
      return overall_X_means, np.sqrt(X_vars)

    if y_stats:
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
      return y_means, np.sqrt(y_vars)
  
  def update_moments(self, X_stats, y_stats):
    """Re-compute statistics of this dataset during transformation"""
    df = self.metadata_df
    self._update_mean_and_std(df, X_stats, y_stats)

  def _update_mean_and_std(self, df, X_stats, y_stats):
    """
    Compute means/stds of X/y from sums/sum_squares of tensors.
    """
    if X_stats:
      X_transform = []
      for _, row in df.iterrows():
        Xt = load_from_disk(os.path.join(self.data_dir, row['X-transformed']))
        Xs = np.sum(Xt,axis=0)
        Xss = np.sum(np.square(Xt),axis=0)
        save_to_disk(Xs, os.path.join(self.data_dir, row['X_sums']))
        save_to_disk(Xss, os.path.join(self.data_dir, row['X_sum_squares']))

    if y_stats:
      y_transform = []
      for _, row in df.iterrows():
        yt = load_from_disk(os.path.join(self.data_dir, row['y-transformed']))
        ys = np.sum(yt,axis=0)
        yss = np.sum(np.square(yt),axis=0)
        save_to_disk(ys, os.path.join(self.data_dir, row['y_sums']))
        save_to_disk(yss, os.path.join(self.data_dir, row['y_sum_squares']))

  def get_grad_statistics(self):
    """Computes and returns statistics of this dataset

    This function assumes that the first task of a dataset holds the energy for
    an input system, and that the remaining tasks holds the gradient for the
    system.

    TODO(rbharath, joegomes): It is unclear whether this should be a Dataset
    function. Might get refactored out.
    TODO(rbharath, joegomes): If y_n were an exposed part of the API, this
    function could be entirely written in userspace.
    """
    if len(self) == 0:
      return None, None, None, None
    df = self.metadata_df
    y = []
    y_n = []
    for _, row in df.iterrows():
      yy = load_from_disk(os.path.join(self.data_dir, row['y']))
      y.append(yy)
      yn = load_from_disk(os.path.join(self.data_dir, row['y_n']))
      y_n.append(np.array(yn))

    y = np.vstack(y)
    y_n = np.sum(y_n, axis=0)

    energy = y[:,0]
    grad = y[:,1:]
    for i in range(energy.size):
      grad[i] *= energy[i]

    ydely_means = np.sum(grad, axis=0)/y_n[1:]

    return grad, ydely_means

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
def convert_df_to_numpy(df, feature_type, tasks, mol_id_field, dtype,
                        verbosity=None):
  """Transforms a dataframe containing deepchem input into numpy arrays"""
  if feature_type not in df.keys():
    raise ValueError(
        "Featurized data does not support requested feature_type %s." % feature_type)
  # perform common train/test split across all tasks
  n_samples = df.shape[0]
  n_tasks = len(tasks)
  ############################################################## TIMING
  time1 = time.time()
  ############################################################## TIMING
  y = np.hstack([
      np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])
  ############################################################## TIMING
  time2 = time.time()
  log("TIMING: convert_df_to_numpy y computation took %0.3f s" % (time2-time1),
      verbosity)
  ############################################################## TIMING
  w = np.ones((n_samples, n_tasks))
  missing = np.zeros_like(y).astype(int)
  feature_shape = None
  ############################################################## TIMING
  time1 = time.time()
  ############################################################## TIMING
  for ind in range(n_samples):
    for task in range(n_tasks):
      if y[ind, task] == "":
        missing[ind, task] = 1
  x_list = list(df[feature_type].values)
  valid_inds = np.array([1 if elt.size > 0 else 0 for elt in x_list], dtype=bool)
  x_list = [elt for (is_valid, elt) in zip(valid_inds, x_list) if is_valid]
  x = np.squeeze(np.array(x_list))
  ############################################################## TIMING
  time2 = time.time()
  log("TIMING: convert_df_to_numpy x computation took %0.3f s" % (time2-time1),
      verbosity)
  ############################################################## TIMING
  sorted_ids = df[mol_id_field].values

  # Set missing data to have weight zero
  ############################################################## TIMING
  time1 = time.time()
  ############################################################## TIMING
  for ind in range(n_samples):
    for task in range(n_tasks):
      if missing[ind, task]:
        y[ind, task] = 0.
        w[ind, task] = 0.
  ############################################################## TIMING
  time2 = time.time()
  log("TIMING: convert_df_to_numpy missing elts computation took %0.3f s"
      % (time2-time1), verbosity)
  ############################################################## TIMING

  sorted_ids = sorted_ids[valid_inds]
  y = y[valid_inds]
  w = w[valid_inds]
  # Adding this assertion in to avoid ill-formed outputs.
  assert len(sorted_ids) == len(x) == len(y) == len(w)
  if dtype == float:
    return sorted_ids, x.astype(float), y.astype(float), w.astype(float)
  elif dtype == object:
    return sorted_ids, x, y.astype(float), w.astype(float)
  else:
    raise ValueError("Unrecognized dtype for featurizer.")
