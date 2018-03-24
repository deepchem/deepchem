"""
Contains wrapper class for datasets.
"""
from __future__ import division
from __future__ import unicode_literals
import json
import os
import math
import numpy as np
import pandas as pd
import random
from deepchem.utils.save import save_to_disk, save_metadata
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import log
from pandas import read_hdf
import tempfile
import time
import shutil
import json
from multiprocessing.dummy import Pool

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"


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
      X_out[start:start + increment] = X_b[:increment]
      start += increment
    return X_out


def pad_batch(batch_size, X_b, y_b, w_b, ids_b):
  """Pads batch to have size precisely batch_size elements.

  Fills in batch by wrapping around samples till whole batch is filled.
  """
  num_samples = len(X_b)
  if num_samples == batch_size:
    return (X_b, y_b, w_b, ids_b)
  # By invariant of when this is called, can assume num_samples > 0
  # and num_samples < batch_size
  if len(X_b.shape) > 1:
    feature_shape = X_b.shape[1:]
    X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
  else:
    X_out = np.zeros((batch_size,), dtype=X_b.dtype)

  if y_b is None:
    y_out = None
  else:
    y_out = np.zeros((batch_size, y_b.shape[1]), dtype=y_b.dtype)

  if w_b is None:
    w_out = None
  else:
    w_out = np.zeros((batch_size, w_b.shape[1]), dtype=w_b.dtype)

  ids_out = np.zeros((batch_size,), dtype=ids_b.dtype)

  # Fill in batch arrays
  start = 0
  # Only the first set of copy will be counted in training loss
  if w_out is not None:
    w_out[start:start + num_samples] = w_b[:]

  while start < batch_size:
    num_left = batch_size - start
    if num_left < num_samples:
      increment = num_left
    else:
      increment = num_samples
    X_out[start:start + increment] = X_b[:increment]

    if y_out is not None:
      y_out[start:start + increment] = y_b[:increment]

    ids_out[start:start + increment] = ids_b[:increment]
    start += increment

  return (X_out, y_out, w_out, ids_out)


class Dataset(object):
  """Abstract base class for datasets defined by X, y, w elements."""

  def __init__(self):
    raise NotImplementedError()

  def __len__(self):
    """
    Get the number of elements in the dataset.
    """
    raise NotImplementedError()

  def get_shape(self):
    """Get the shape of the dataset.

    Returns four tuples, giving the shape of the X, y, w, and ids arrays.
    """
    raise NotImplementedError()

  def get_task_names(self):
    """Get the names of the tasks associated with this dataset."""
    raise NotImplementedError()

  @property
  def X(self):
    """Get the X vector for this dataset as a single numpy array."""
    raise NotImplementedError()

  @property
  def y(self):
    """Get the y vector for this dataset as a single numpy array."""
    raise NotImplementedError()

  @property
  def ids(self):
    """Get the ids vector for this dataset as a single numpy array."""

    raise NotImplementedError()

  @property
  def w(self):
    """Get the weight vector for this dataset as a single numpy array."""
    raise NotImplementedError()

  def iterbatches(self,
                  batch_size=None,
                  epoch=0,
                  deterministic=False,
                  pad_batches=False):
    """

    Parameters
    ----------


    Returns
    -------

    """
    """Get an object that iterates over minibatches from the dataset.

    Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).
    """
    raise NotImplementedError()

  def itersamples(self):
    """Get an object that iterates over the samples in the dataset.

    Example:

    >>> dataset = NumpyDataset(np.ones((2,2)))
    >>> for x, y, w, id in dataset.itersamples():
    ...   print(x.tolist(), y.tolist(), w.tolist(), id)
    [1.0, 1.0] [0.0] [0.0] 0
    [1.0, 1.0] [0.0] [0.0] 1
    """
    raise NotImplementedError()

  def transform(self, fn, **args):
    """Construct a new dataset by applying a transformation to every sample in this dataset.

    The argument is a function that can be called as follows:

    >> newx, newy, neww = fn(x, y, w)

    It might be called only once with the whole dataset, or multiple times with
    different subsets of the data.  Each time it is called, it should transform
    the samples and return the transformed data.

    Parameters
    ----------
    fn: function
      A function to apply to each sample in the dataset

    Returns
    -------
    a newly constructed Dataset object
    """
    raise NotImplementedError()

  def get_statistics(self, X_stats=True, y_stats=True):
    """Compute and return statistics of this dataset."""
    X_means = 0.0
    X_m2 = 0.0
    y_means = 0.0
    y_m2 = 0.0
    n = 0
    for X, y, _, _ in self.itersamples():
      n += 1
      if X_stats:
        dx = X - X_means
        X_means += dx / n
        X_m2 += dx * (X - X_means)
      if y_stats:
        dy = y - y_means
        y_means += dy / n
        y_m2 += dy * (y - y_means)
    if n < 2:
      X_stds = 0.0
      y_stds = 0
    else:
      X_stds = np.sqrt(X_m2 / n)
      y_stds = np.sqrt(y_m2 / n)
    if X_stats and not y_stats:
      return X_means, X_stds
    elif y_stats and not X_stats:
      return y_means, y_stds
    elif X_stats and y_stats:
      return X_means, X_stds, y_means, y_stds
    else:
      return None

  def make_iterator(self,
                    batch_size=100,
                    epochs=1,
                    deterministic=False,
                    pad_batches=False):
    """Create a tf.data.Iterator that iterates over the data in this Dataset.

    The iterator's get_next() method returns a tuple of three tensors (X, y, w)
    which can be used to retrieve the features, labels, and weights respectively.

    Parameters
    ----------
    batch_size: int
      the number of samples to include in each batch
    epochs: int
      the number of times to iterate over the Dataset
    deterministic: bool
      if True, the data is produced in order.  If False, a different random
      permutation of the data is used for each epoch.
    pad_batches: bool
      if True, batches are padded as necessary to make the size of each batch
      exactly equal batch_size.
    """
    # Retrieve the first sample so we can determine the dtypes.

    import tensorflow as tf
    X, y, w, ids = next(self.itersamples())
    dtypes = (tf.as_dtype(X.dtype), tf.as_dtype(y.dtype), tf.as_dtype(w.dtype))
    shapes = (tf.TensorShape([None] + list(X.shape)),
              tf.TensorShape([None] + list(y.shape)),
              tf.TensorShape([None] + list(w.shape)))

    # Create a Tensorflow Dataset and have it create an Iterator.

    def gen_data():
      for epoch in range(epochs):
        for X, y, w, ids in self.iterbatches(batch_size, epoch, deterministic,
                                             pad_batches):
          yield (X, y, w)

    dataset = tf.data.Dataset.from_generator(gen_data, dtypes, shapes)
    return dataset.make_one_shot_iterator()


class NumpyDataset(Dataset):
  """A Dataset defined by in-memory numpy arrays."""

  def __init__(self, X, y=None, w=None, ids=None, n_tasks=1):
    n_samples = len(X)
    # The -1 indicates that y will be reshaped to have length -1
    if n_samples > 0:
      if y is not None:
        y = np.reshape(y, (n_samples, -1))
        if w is not None:
          w = np.reshape(w, (n_samples, -1))
      else:
        # Set labels to be zero, with zero weights
        y = np.zeros((n_samples, n_tasks))
        w = np.zeros_like(y)
    if ids is None:
      ids = np.arange(n_samples)
    if w is None:
      w = np.ones_like(y)
    self._X = X
    self._y = y
    self._w = w
    self._ids = np.array(ids, dtype=object)

  def __len__(self):
    """
    Get the number of elements in the dataset.
    """
    return len(self._y)

  def get_shape(self):
    """Get the shape of the dataset.

    Returns four tuples, giving the shape of the X, y, w, and ids arrays.
    """
    return self._X.shape, self._y.shape, self._w.shape, self._ids.shape

  def get_task_names(self):
    """Get the names of the tasks associated with this dataset."""
    return np.arange(self._y.shape[1])

  @property
  def X(self):
    """Get the X vector for this dataset as a single numpy array."""
    return self._X

  @property
  def y(self):
    """Get the y vector for this dataset as a single numpy array."""
    return self._y

  @property
  def ids(self):
    """Get the ids vector for this dataset as a single numpy array."""
    return self._ids

  @property
  def w(self):
    """Get the weight vector for this dataset as a single numpy array."""
    return self._w

  def iterbatches(self,
                  batch_size=None,
                  epoch=0,
                  deterministic=False,
                  pad_batches=False):
    """Get an object that iterates over minibatches from the dataset.

    Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).
    """

    def iterate(dataset, batch_size, deterministic, pad_batches):
      n_samples = dataset._X.shape[0]
      if not deterministic:
        sample_perm = np.random.permutation(n_samples)
      else:
        sample_perm = np.arange(n_samples)
      if batch_size is None:
        batch_size = n_samples
      batch_idx = 0
      num_batches = np.math.ceil(n_samples / batch_size)
      while batch_idx < num_batches:
        start = batch_idx * batch_size
        end = min(n_samples, (batch_idx + 1) * batch_size)
        indices = range(start, end)
        perm_indices = sample_perm[indices]
        X_batch = dataset._X[perm_indices]
        y_batch = dataset._y[perm_indices]
        w_batch = dataset._w[perm_indices]
        ids_batch = dataset._ids[perm_indices]
        if pad_batches:
          (X_batch, y_batch, w_batch, ids_batch) = pad_batch(
              batch_size, X_batch, y_batch, w_batch, ids_batch)
        batch_idx += 1
        yield (X_batch, y_batch, w_batch, ids_batch)

    return iterate(self, batch_size, deterministic, pad_batches)

  def itersamples(self):
    """Get an object that iterates over the samples in the dataset.

    Example:

    >>> dataset = NumpyDataset(np.ones((2,2)))
    >>> for x, y, w, id in dataset.itersamples():
    ...   print(x.tolist(), y.tolist(), w.tolist(), id)
    [1.0, 1.0] [0.0] [0.0] 0
    [1.0, 1.0] [0.0] [0.0] 1
    """
    n_samples = self._X.shape[0]
    return ((self._X[i], self._y[i], self._w[i], self._ids[i])
            for i in range(n_samples))

  def transform(self, fn, **args):
    """Construct a new dataset by applying a transformation to every sample in this dataset.

    The argument is a function that can be called as follows:

    >> newx, newy, neww = fn(x, y, w)

    It might be called only once with the whole dataset, or multiple times with
    different subsets of the data.  Each time it is called, it should transform
    the samples and return the transformed data.

    Parameters
    ----------
    fn: function
      A function to apply to each sample in the dataset

    Returns
    -------
    a newly constructed Dataset object
    """
    newx, newy, neww = fn(self._X, self._y, self._w)
    return NumpyDataset(newx, newy, neww, self._ids[:])

  def select(self, indices, select_dir=None):
    """Creates a new dataset from a selection of indices from self.

    TODO(rbharath): select_dir is here due to dc.splits always passing in
    splits.

    Parameters
    ----------
    indices: list
      List of indices to select.
    select_dir: string
      Ignored.
    """
    X = self.X[indices]
    y = self.y[indices]
    w = self.w[indices]
    ids = self.ids[indices]
    return NumpyDataset(X, y, w, ids)

  @staticmethod
  def from_DiskDataset(ds):
    """

    Parameters
    ----------
    ds : DiskDataset
    DiskDataset to transorm to NumpyDataset

    Returns
    -------
    NumpyDataset
      Data of ds as NumpyDataset

    """
    return NumpyDataset(ds.X, ds.y, ds.w, ds.ids)

  @staticmethod
  def to_json(self, fname):
    d = {
        'X': self.X.tolist(),
        'y': self.y.tolist(),
        'w': self.w.tolist(),
        'ids': self.ids.tolist()
    }
    with open(fname, 'w') as fout:
      json.dump(d, fout)

  @staticmethod
  def from_json(fname):
    with open(fname) as fin:
      d = json.load(fin)
      return NumpyDataset(d['X'], d['y'], d['w'], d['ids'])

  @staticmethod
  def merge(datasets):
    """
    Parameters
    ----------
    datasets: list of deepchem.data.NumpyDataset
      list of datasets to merge

    Returns
    -------
    Single deepchem.data.NumpyDataset with data concatenated over axis 0
    """
    X, y, w, ids = datasets[0].X, datasets[0].y, datasets[0].w, datasets[0].ids
    for dataset in datasets[1:]:
      X = np.concatenate([X, dataset.X], axis=0)
      y = np.concatenate([y, dataset.y], axis=0)
      w = np.concatenate([w, dataset.w], axis=0)
      ids = np.concatenate(
          [ids, dataset.ids],
          axis=0,
      )

    return NumpyDataset(X, y, w, ids, n_tasks=y.shape[1])


class DiskDataset(Dataset):
  """
  A Dataset that is stored as a set of files on disk.
  """

  def __init__(self, data_dir, verbose=True):
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    self.data_dir = data_dir
    self.verbose = verbose

    log("Loading dataset from disk.", self.verbose)
    self.tasks, self.metadata_df = self.load_metadata()

  @staticmethod
  def create_dataset(shard_generator, data_dir=None, tasks=[], verbose=True):
    """Creates a new DiskDataset

    Parameters
    ----------
    shard_generator: Iterable
      An iterable (either a list or generator) that provides tuples of data
      (X, y, w, ids). Each tuple will be written to a separate shard on disk.
    data_dir: str
      Filename for data directory. Creates a temp directory if none specified.
    tasks: list
      List of tasks for this dataset.
    """
    if data_dir is None:
      data_dir = tempfile.mkdtemp()
    elif not os.path.exists(data_dir):
      os.makedirs(data_dir)

    metadata_rows = []
    time1 = time.time()
    for shard_num, (X, y, w, ids) in enumerate(shard_generator):
      basename = "shard-%d" % shard_num
      metadata_rows.append(
          DiskDataset.write_data_to_disk(data_dir, basename, tasks, X, y, w,
                                         ids))
    metadata_df = DiskDataset._construct_metadata(metadata_rows)
    save_metadata(tasks, metadata_df, data_dir)
    time2 = time.time()
    log("TIMING: dataset construction took %0.3f s" % (time2 - time1), verbose)
    return DiskDataset(data_dir, verbose=verbose)

  def load_metadata(self):
    try:
      tasks_filename, metadata_filename = self._get_metadata_filename()
      with open(tasks_filename) as fin:
        tasks = json.load(fin)
      metadata_df = pd.read_csv(metadata_filename, compression='gzip')
      metadata_df = metadata_df.where((pd.notnull(metadata_df)), None)
      return tasks, metadata_df
    except Exception as e:
      pass

    # Load obsolete format -> save in new format
    metadata_filename = os.path.join(self.data_dir, "metadata.joblib")
    if os.path.exists(metadata_filename):
      tasks, metadata_df = load_from_disk(metadata_filename)
      del metadata_df['task_names']
      del metadata_df['basename']
      save_metadata(tasks, metadata_df, self.data_dir)
      return tasks, metadata_df
    raise ValueError("No Metadata Found On Disk")

  @staticmethod
  def _construct_metadata(metadata_entries):
    """Construct a dataframe containing metadata.

    metadata_entries should have elements returned by write_data_to_disk
    above.
    """
    columns = ('ids', 'X', 'y', 'w')
    metadata_df = pd.DataFrame(metadata_entries, columns=columns)
    return metadata_df

  @staticmethod
  def write_data_to_disk(data_dir,
                         basename,
                         tasks,
                         X=None,
                         y=None,
                         w=None,
                         ids=None):
    if X is not None:
      out_X = "%s-X.joblib" % basename
      save_to_disk(X, os.path.join(data_dir, out_X))
    else:
      out_X = None

    if y is not None:
      out_y = "%s-y.joblib" % basename
      save_to_disk(y, os.path.join(data_dir, out_y))
    else:
      out_y = None

    if w is not None:
      out_w = "%s-w.joblib" % basename
      save_to_disk(w, os.path.join(data_dir, out_w))
    else:
      out_w = None

    if ids is not None:
      out_ids = "%s-ids.joblib" % basename
      save_to_disk(ids, os.path.join(data_dir, out_ids))
    else:
      out_ids = None

    # note that this corresponds to the _construct_metadata column order
    return [out_ids, out_X, out_y, out_w]

  def save_to_disk(self):
    """Save dataset to disk."""
    save_metadata(self.tasks, self.metadata_df, self.data_dir)

  def move(self, new_data_dir):
    """Moves dataset to new directory."""
    shutil.move(self.data_dir, new_data_dir)
    self.data_dir = new_data_dir

  def get_task_names(self):
    """
    Gets learning tasks associated with this dataset.
    """
    return self.tasks
    # if not len(self.metadata_df):
    #  raise ValueError("No data in dataset.")
    # return next(self.metadata_df.iterrows())[1]['task_names']

  def reshard(self, shard_size):
    """Reshards data to have specified shard size."""
    # Create temp directory to store resharded version
    reshard_dir = tempfile.mkdtemp()
    new_metadata = []

    # Write data in new shards
    def generator():
      tasks = self.get_task_names()
      X_next = np.zeros((0,) + self.get_data_shape())
      y_next = np.zeros((0,) + (len(tasks),))
      w_next = np.zeros((0,) + (len(tasks),))
      ids_next = np.zeros((0,), dtype=object)
      for (X, y, w, ids) in self.itershards():
        X_next = np.concatenate([X_next, X], axis=0)
        y_next = np.concatenate([y_next, y], axis=0)
        w_next = np.concatenate([w_next, w], axis=0)
        ids_next = np.concatenate([ids_next, ids])
        while len(X_next) > shard_size:
          X_batch, X_next = X_next[:shard_size], X_next[shard_size:]
          y_batch, y_next = y_next[:shard_size], y_next[shard_size:]
          w_batch, w_next = w_next[:shard_size], w_next[shard_size:]
          ids_batch, ids_next = ids_next[:shard_size], ids_next[shard_size:]
          yield (X_batch, y_batch, w_batch, ids_batch)
      # Handle spillover from last shard
      yield (X_next, y_next, w_next, ids_next)

    resharded_dataset = DiskDataset.create_dataset(
        generator(), data_dir=reshard_dir, tasks=self.tasks)
    shutil.rmtree(self.data_dir)
    shutil.move(reshard_dir, self.data_dir)
    self.metadata_df = resharded_dataset.metadata_df
    self.save_to_disk()

  def get_data_shape(self):
    """
    Gets array shape of datapoints in this dataset.
    """
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_X = load_from_disk(
        os.path.join(self.data_dir,
                     next(self.metadata_df.iterrows())[1]['X']))
    return np.shape(sample_X)[1:]

  def get_shard_size(self):
    """Gets size of shards on disk."""
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_y = load_from_disk(
        os.path.join(self.data_dir,
                     next(self.metadata_df.iterrows())[1]['y']))
    return len(sample_y)

  def _get_metadata_filename(self):
    """
    Get standard location for metadata file.
    """
    metadata_filename = os.path.join(self.data_dir, "metadata.csv.gzip")
    tasks_filename = os.path.join(self.data_dir, "tasks.json")
    return tasks_filename, metadata_filename

  def get_number_shards(self):
    """
    Returns the number of shards for this dataset.
    """
    return self.metadata_df.shape[0]

  def itershards(self):
    """
    Return an object that iterates over all shards in dataset.

    Datasets are stored in sharded fashion on disk. Each call to next() for the
    generator defined by this function returns the data from a particular shard.
    The order of shards returned is guaranteed to remain fixed.
    """

    def iterate(dataset):
      for _, row in dataset.metadata_df.iterrows():
        X = np.array(load_from_disk(os.path.join(dataset.data_dir, row['X'])))
        ids = np.array(
            load_from_disk(os.path.join(dataset.data_dir, row['ids'])),
            dtype=object)
        # These columns may be missing is the dataset is unlabelled.
        if row['y'] is not None:
          y = np.array(load_from_disk(os.path.join(dataset.data_dir, row['y'])))
        else:
          y = None
        if row['w'] is not None:
          w_filename = os.path.join(dataset.data_dir, row['w'])
          if os.path.exists(w_filename):
            w = np.array(load_from_disk(w_filename))
          else:
            w = np.ones(y.shape)
        else:
          w = None
        yield (X, y, w, ids)

    return iterate(self)

  def iterbatches(self,
                  batch_size=None,
                  epoch=0,
                  deterministic=False,
                  pad_batches=False):
    """ Get an object that iterates over minibatches from the dataset. It is guaranteed
    that the number of batches returned is math.ceil(len(dataset)/batch_size).

    Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).


    Parameters:
    -----------
    batch_size: int
      Number of elements in a batch. If None, then it yields batches with size equal to the size
      of each individual shard.

    epoch: int
      Not used

    deterministic: bool
      Whether or not we should should shuffle each shard before generating the batches.
      Note that this is only local in the sense that it does not ever mix between different
      shards.

    pad_batches: bool
      Whether or not we should pad the last batch, globally, such that it has exactly batch_size
      elements.


    """

    def iterate(dataset, batch_size):
      num_shards = dataset.get_number_shards()
      if not deterministic:
        shard_perm = np.random.permutation(num_shards)
      else:
        shard_perm = np.arange(num_shards)

      # (ytz): Depending on the application, thread-based pools may be faster
      # than process based pools, since process based pools need to pickle/serialize
      # objects as an extra overhead. Also, as hideously as un-thread safe this looks,
      # we're actually protected by the GIL.
      pool = Pool(1)  # mp.dummy aliases ThreadPool to Pool
      next_shard = pool.apply_async(dataset.get_shard, (shard_perm[0],))

      total_yield = 0

      if batch_size is None:
        num_global_batches = num_shards
      else:
        num_global_batches = math.ceil(dataset.get_shape()[0][0] / batch_size)

      cur_global_batch = 0
      cur_shard = 0
      carry = None

      while cur_global_batch < num_global_batches:

        X, y, w, ids = next_shard.get()
        if cur_shard < num_shards - 1:
          next_shard = pool.apply_async(dataset.get_shard,
                                        (shard_perm[cur_shard + 1],))
        else:
          pool.close()

        if carry is not None:
          X = np.concatenate([carry[0], X], axis=0)
          if y is not None:
            y = np.concatenate([carry[1], y], axis=0)
          if w is not None:
            w = np.concatenate([carry[2], w], axis=0)
          ids = np.concatenate([carry[3], ids], axis=0)
          carry = None

        n_shard_samples = X.shape[0]
        cur_local_batch = 0
        if batch_size is None:
          shard_batch_size = n_shard_samples
        else:
          shard_batch_size = batch_size

        if n_shard_samples == 0:
          cur_shard += 1
          if batch_size is None:
            cur_global_batch += 1
          continue

        num_local_batches = math.ceil(n_shard_samples / shard_batch_size)
        if not deterministic:
          sample_perm = np.random.permutation(n_shard_samples)
        else:
          sample_perm = np.arange(n_shard_samples)

        while cur_local_batch < num_local_batches:
          start = cur_local_batch * shard_batch_size
          end = min(n_shard_samples, (cur_local_batch + 1) * shard_batch_size)

          indices = range(start, end)
          perm_indices = sample_perm[indices]
          X_b = X[perm_indices]

          if y is not None:
            y_b = y[perm_indices]
          else:
            y_b = None

          if w is not None:
            w_b = w[perm_indices]
          else:
            w_b = None

          ids_b = ids[perm_indices]

          assert len(X_b) <= shard_batch_size
          if len(X_b) < shard_batch_size and cur_shard != num_shards - 1:
            assert carry is None
            carry = [X_b, y_b, w_b, ids_b]
          else:

            # (ytz): this skips everything except possibly the last shard
            if pad_batches:
              (X_b, y_b, w_b, ids_b) = pad_batch(shard_batch_size, X_b, y_b,
                                                 w_b, ids_b)

            yield X_b, y_b, w_b, ids_b
            cur_global_batch += 1
          cur_local_batch += 1
        cur_shard += 1

    return iterate(self, batch_size)

  def itersamples(self):
    """Get an object that iterates over the samples in the dataset.

    Example:

    >>> dataset = DiskDataset.from_numpy(np.ones((2,2)), np.ones((2,1)), verbose=False)
    >>> for x, y, w, id in dataset.itersamples():
    ...   print(x.tolist(), y.tolist(), w.tolist(), id)
    [1.0, 1.0] [1.0] [1.0] 0
    [1.0, 1.0] [1.0] [1.0] 1
    """

    def iterate(dataset):
      for (X_shard, y_shard, w_shard, ids_shard) in dataset.itershards():
        n_samples = X_shard.shape[0]
        for i in range(n_samples):

          def sanitize(elem):
            if elem is None:
              return None
            else:
              return elem[i]

          yield map(sanitize, [X_shard, y_shard, w_shard, ids_shard])

    return iterate(self)

  def transform(self, fn, **args):
    """Construct a new dataset by applying a transformation to every sample in this dataset.

    The argument is a function that can be called as follows:

    >> newx, newy, neww = fn(x, y, w)

    It might be called only once with the whole dataset, or multiple times with different
    subsets of the data.  Each time it is called, it should transform the samples and return
    the transformed data.

    Parameters
    ----------
    fn: function
      A function to apply to each sample in the dataset
    out_dir: string
      The directory to save the new dataset in.  If this is omitted, a temporary directory
      is created automatically

    Returns
    -------
    a newly constructed Dataset object
    """
    if 'out_dir' in args:
      out_dir = args['out_dir']
    else:
      out_dir = tempfile.mkdtemp()
    tasks = self.get_task_names()

    def generator():
      for shard_num, row in self.metadata_df.iterrows():
        X, y, w, ids = self.get_shard(shard_num)
        newx, newy, neww = fn(X, y, w)
        yield (newx, newy, neww, ids)

    return DiskDataset.create_dataset(
        generator(), data_dir=out_dir, tasks=tasks)

  @staticmethod
  def from_numpy(X,
                 y,
                 w=None,
                 ids=None,
                 tasks=None,
                 data_dir=None,
                 verbose=True):
    """Creates a DiskDataset object from specified Numpy arrays."""
    # if data_dir is None:
    #  data_dir = tempfile.mkdtemp()
    n_samples = len(X)
    # The -1 indicates that y will be reshaped to have length -1
    if n_samples > 0:
      y = np.reshape(y, (n_samples, -1))
      if w is not None:
        w = np.reshape(w, (n_samples, -1))
    if ids is None:
      ids = np.arange(n_samples)
    if w is None:
      w = np.ones_like(y)
    if tasks is None:
      n_tasks = y.shape[1]
      tasks = np.arange(n_tasks)
    # raw_data = (X, y, w, ids)
    return DiskDataset.create_dataset(
        [(X, y, w, ids)], data_dir=data_dir, tasks=tasks, verbose=verbose)

  @staticmethod
  def merge(datasets, merge_dir=None):
    """Merges provided datasets into a merged dataset."""
    if merge_dir is not None:
      if not os.path.exists(merge_dir):
        os.makedirs(merge_dir)
    else:
      merge_dir = tempfile.mkdtemp()

    # Protect against generator exhaustion
    datasets = list(datasets)

    # This ensures tasks are consistent for all datasets
    tasks = []
    for dataset in datasets:
      try:
        tasks.append(dataset.tasks)
      except AttributeError:
        pass
    if tasks:
      if len(tasks) < len(datasets) or len(set(map(tuple, tasks))) > 1:
        raise ValueError(
            'Cannot merge datasets with different task specifications')
      tasks = tasks[0]

    def generator():
      for ind, dataset in enumerate(datasets):
        X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
        yield (X, y, w, ids)

    return DiskDataset.create_dataset(
        generator(), data_dir=merge_dir, tasks=tasks)

  def subset(self, shard_nums, subset_dir=None):
    """Creates a subset of the original dataset on disk."""
    if subset_dir is not None:
      if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    else:
      subset_dir = tempfile.mkdtemp()
    tasks = self.get_task_names()

    def generator():
      for shard_num, row in self.metadata_df.iterrows():
        if shard_num not in shard_nums:
          continue
        X, y, w, ids = self.get_shard(shard_num)
        yield (X, y, w, ids)

    return DiskDataset.create_dataset(
        generator(), data_dir=subset_dir, tasks=tasks)

  def sparse_shuffle(self):
    """Shuffling that exploits data sparsity to shuffle large datasets.

    Only for 1-dimensional feature vectors (does not work for tensorial
    featurizations).
    """
    time1 = time.time()
    shard_size = self.get_shard_size()
    num_shards = self.get_number_shards()
    X_sparses, ys, ws, ids = [], [], [], []
    num_features = None
    for i in range(num_shards):
      (X_s, y_s, w_s, ids_s) = self.get_shard(i)
      if num_features is None:
        num_features = X_s.shape[1]
      X_sparse = sparsify_features(X_s)
      X_sparses, ys, ws, ids = (X_sparses + [X_sparse], ys + [y_s], ws + [w_s],
                                ids + [np.atleast_1d(np.squeeze(ids_s))])
    # Get full dataset in memory
    (X_sparse, y, w, ids) = (np.vstack(X_sparses), np.vstack(ys), np.vstack(ws),
                             np.concatenate(ids))
    # Shuffle in memory
    num_samples = len(X_sparse)
    permutation = np.random.permutation(num_samples)
    X_sparse, y, w, ids = (X_sparse[permutation], y[permutation],
                           w[permutation], ids[permutation])
    # Write shuffled shards out to disk
    for i in range(num_shards):
      start, stop = i * shard_size, (i + 1) * shard_size
      (X_sparse_s, y_s, w_s, ids_s) = (X_sparse[start:stop], y[start:stop],
                                       w[start:stop], ids[start:stop])
      X_s = densify_features(X_sparse_s, num_features)
      self.set_shard(i, X_s, y_s, w_s, ids_s)
    time2 = time.time()
    log("TIMING: sparse_shuffle took %0.3f s" % (time2 - time1), self.verbose)

  def complete_shuffle(self, data_dir=None):
    """
    Completely shuffle across all data, across all shards.

    Note: this loads all the data into ram, and can be prohibitively
    expensive for larger datasets.

    Parameters
    ----------
    shard_size: int
      size of the resulting dataset's size. If None, then the first
      shard's shard_size will be used.

    Returns
    -------
    DiskDatasset
      A DiskDataset with a single shard.

    """
    all_X = []
    all_y = []
    all_w = []
    all_ids = []
    for Xs, ys, ws, ids in self.itershards():
      all_X.append(Xs)
      if ys is not None:
        all_y.append(ys)
      if ws is not None:
        all_w.append(ws)
      all_ids.append(ids)

    all_X = np.concatenate(all_X)
    all_y = np.concatenate(all_y)
    all_w = np.concatenate(all_w)
    all_ids = np.concatenate(all_ids)

    perm = np.random.permutation(all_X.shape[0])
    all_X = all_X[perm]
    all_y = all_y[perm]
    all_w = all_w[perm]
    all_ids = all_ids[perm]

    return DiskDataset.from_numpy(
        all_X, all_y, all_w, all_ids, data_dir=data_dir)

  def shuffle_each_shard(self):
    """Shuffles elements within each shard of the datset."""
    tasks = self.get_task_names()
    # Shuffle the arrays corresponding to each row in metadata_df
    n_rows = len(self.metadata_df.index)
    n_rows = len(self.metadata_df.index)
    for i in range(n_rows):
      row = self.metadata_df.iloc[i]
      X, y, w, ids = self.get_shard(i)
      n = X.shape[0]
      permutation = np.random.permutation(n)
      X, y, w, ids = (X[permutation], y[permutation], w[permutation],
                      ids[permutation])
      DiskDataset.write_data_to_disk(self.data_dir, "", tasks, X, y, w, ids)

  def shuffle_shards(self):
    """Shuffles the order of the shards for this dataset."""
    metadata_rows = self.metadata_df.values.tolist()
    random.shuffle(metadata_rows)
    self.metadata_df = DiskDataset._construct_metadata(metadata_rows)
    self.save_to_disk()

  def get_shard(self, i):
    """Retrieves data for the i-th shard from disk."""
    row = self.metadata_df.iloc[i]
    X = np.array(load_from_disk(os.path.join(self.data_dir, row['X'])))

    if row['y'] is not None:
      y = np.array(load_from_disk(os.path.join(self.data_dir, row['y'])))
    else:
      y = None

    if row['w'] is not None:
      # TODO (ytz): Under what condition does this exist but the file itself doesn't?
      w_filename = os.path.join(self.data_dir, row['w'])
      if os.path.exists(w_filename):
        w = np.array(load_from_disk(w_filename))
      else:
        w = np.ones(y.shape)
    else:
      w = None

    ids = np.array(
        load_from_disk(os.path.join(self.data_dir, row['ids'])), dtype=object)
    return (X, y, w, ids)

  def add_shard(self, X, y, w, ids):
    """Adds a data shard."""
    metadata_rows = self.metadata_df.values.tolist()
    shard_num = len(metadata_rows)
    basename = "shard-%d" % shard_num
    tasks = self.get_task_names()
    metadata_rows.append(
        DiskDataset.write_data_to_disk(self.data_dir, basename, tasks, X, y, w,
                                       ids))
    self.metadata_df = DiskDataset._construct_metadata(metadata_rows)
    self.save_to_disk()

  def set_shard(self, shard_num, X, y, w, ids):
    """Writes data shard to disk"""
    basename = "shard-%d" % shard_num
    tasks = self.get_task_names()
    DiskDataset.write_data_to_disk(self.data_dir, basename, tasks, X, y, w, ids)

  def select(self, indices, select_dir=None):
    """Creates a new dataset from a selection of indices from self.

    Parameters
    ----------
    select_dir: string
      Path to new directory that the selected indices will be copied to.
    indices: list
      List of indices to select.
    """
    if select_dir is not None:
      if not os.path.exists(select_dir):
        os.makedirs(select_dir)
    else:
      select_dir = tempfile.mkdtemp()
    # Handle edge case with empty indices
    if not len(indices):
      return DiskDataset.create_dataset(
          [], data_dir=select_dir, verbose=self.verbose)
    indices = np.array(sorted(indices)).astype(int)
    tasks = self.get_task_names()

    def generator():
      count, indices_count = 0, 0
      for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
        shard_len = len(X)
        # Find indices which rest in this shard
        num_shard_elts = 0
        while indices[indices_count + num_shard_elts] < count + shard_len:
          num_shard_elts += 1
          if indices_count + num_shard_elts >= len(indices):
            break
        # Need to offset indices to fit within shard_size
        shard_inds = indices[indices_count:
                             indices_count + num_shard_elts] - count
        X_sel = X[shard_inds]
        # Handle the case of datasets with y/w missing
        if y is not None:
          y_sel = y[shard_inds]
        else:
          y_sel = None
        if w is not None:
          w_sel = w[shard_inds]
        else:
          w_sel = None
        ids_sel = ids[shard_inds]
        yield (X_sel, y_sel, w_sel, ids_sel)
        # Updating counts
        indices_count += num_shard_elts
        count += shard_len
        # Break when all indices have been used up already
        if indices_count >= len(indices):
          return

    return DiskDataset.create_dataset(
        generator(), data_dir=select_dir, tasks=tasks, verbose=self.verbose)

  @property
  def ids(self):
    """Get the ids vector for this dataset as a single numpy array."""
    if len(self) == 0:
      return np.array([])
    ids = []
    for (_, _, _, ids_b) in self.itershards():
      ids.append(np.atleast_1d(np.squeeze(ids_b)))
    return np.concatenate(ids)

  @property
  def X(self):
    """Get the X vector for this dataset as a single numpy array."""
    Xs = []
    one_dimensional = False
    for (X_b, _, _, _) in self.itershards():
      Xs.append(X_b)
      if len(X_b.shape) == 1:
        one_dimensional = True
    if not one_dimensional:
      return np.vstack(Xs)
    else:
      return np.concatenate(Xs)

  @property
  def y(self):
    """Get the y vector for this dataset as a single numpy array."""
    ys = []
    for (_, y_b, _, _) in self.itershards():
      ys.append(y_b)
    return np.vstack(ys)

  @property
  def w(self):
    """Get the weight vector for this dataset as a single numpy array."""
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
      y = load_from_disk(os.path.join(self.data_dir, row['ids']))
      total += len(y)
    return total

  def get_shape(self):
    """Finds shape of dataset."""
    n_tasks = len(self.get_task_names())
    X_shape = np.array((0,) + (0,) * len(self.get_data_shape()))
    ids_shape = np.array((0,))
    if n_tasks > 0:
      y_shape = np.array((0,) + (0,))
      w_shape = np.array((0,) + (0,))
    else:
      y_shape = tuple()
      w_shape = tuple()

    for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
      if shard_num == 0:
        X_shape += np.array(X.shape)
        if n_tasks > 0:
          y_shape += np.array(y.shape)
          w_shape += np.array(w.shape)
        ids_shape += np.array(ids.shape)
      else:
        X_shape[0] += np.array(X.shape)[0]
        if n_tasks > 0:
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


class Databag(object):
  """
  A utility class to iterate through multiple datasets together.
  """

  def __init__(self, datasets=None):
    if datasets is None:
      self.datasets = dict()
    else:
      self.datasets = datasets

  def add_dataset(self, key, dataset):
    self.datasets[key] = dataset

  def iterbatches(self, **kwargs):
    """
    Loop through all internal datasets in the same order
    Parameters
    ----------
    batch_size: int
      Number of samples from each dataset to return
    epoch: int
      Number of times to loop through the datasets
    pad_batches: boolean
      Should all batches==batch_size

    Returns
    -------
    Generator which yields a dictionary {key: dataset.X[batch]}

    """
    key_order = [x for x in self.datasets.keys()]
    if "epochs" in kwargs:
      epochs = kwargs['epochs']
      del kwargs['epochs']
    else:
      epochs = 1
    kwargs['deterministic'] = True
    for epoch in range(epochs):
      iterators = [self.datasets[x].iterbatches(**kwargs) for x in key_order]
      for tup in zip(*iterators):
        m_d = {key_order[i]: tup[i][0] for i in range(len(key_order))}
        yield m_d
