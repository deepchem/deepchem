"""
Contains wrapper class for datasets.
"""
import json
import os
import math
import deepchem as dc
import numpy as np
import pandas as pd
import random
import logging
from pandas import read_hdf
import tempfile
import time
import shutil
import json
import warnings
import multiprocessing
from deepchem.utils.save import save_to_disk, save_metadata
from deepchem.utils.save import load_from_disk

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import OneOrMany, Shape

Batch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

logger = logging.getLogger(__name__)


def sparsify_features(X: np.ndarray) -> np.ndarray:
  """Extracts a sparse feature representation from dense feature array.

  Parameters
  ----------
  X: np.ndarray
    Of shape `(n_samples, ...)

  Returns
  -------
  X_sparse, a np.ndarray with `dtype=object` where `X_sparse[i]` is a
  typle of `(nonzero_inds, nonzero_vals)` with nonzero indices and
  values in the i-th sample of `X`.
  """
  n_samples = len(X)
  X_sparse = []
  for i in range(n_samples):
    nonzero_inds = np.nonzero(X[i])[0]
    nonzero_vals = X[i][nonzero_inds]
    X_sparse.append((nonzero_inds, nonzero_vals))
  X_sparse = np.array(X_sparse, dtype=object)
  return X_sparse


def densify_features(X_sparse: np.ndarray, num_features: int) -> np.ndarray:
  """Expands sparse feature representation to dense feature array.

  Assumes that the sparse representation was constructed from an array
  which had original shape `(n_samples, num_features)` so doesn't
  support reconstructing multidimensional dense arrays.

  Parameters
  ----------
  X_sparse: np.ndarray
    Must have `dtype=object`. `X_sparse[i]` must be a tuple of nonzero
    indices and values.
  num_features: int
    Number of features in dense array.

  Returns
  -------
  X, a np.ndarray of shape `(n_samples, num_features)`.
  """
  n_samples = len(X_sparse)
  X = np.zeros((n_samples, num_features))
  for i in range(n_samples):
    nonzero_inds, nonzero_vals = X_sparse[i]
    X[i][nonzero_inds.astype(int)] = nonzero_vals
  return X


def pad_features(batch_size: int, X_b: np.ndarray) -> np.ndarray:
  """Pads a batch of features to have precisely batch_size elements.

  Given an array of features with length less than or equal to
  batch-size, pads it to `batch_size` length. It does this by
  repeating the original features in tiled fashion. For illustration,
  suppose that `len(X_b) == 3` and `batch_size == 10`.

  >>> X_b = np.arange(3)
  >>> X_b
  array([0, 1, 2])
  >>> batch_size = 10
  >>> X_manual = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
  >>> X_out = pad_features(batch_size, X_b)
  >>> assert (X_manual == X_out).all()

  This function is similar to `pad_batch` but doesn't handle labels
  `y` or weights `w` and is intended to be used for inference-time
  query processing.

  Parameters
  ----------
  batch_size: int
    The number of datapoints in a batch
  X_b: np.ndarray
    Must be such that `len(X_b) <= batch_size`

  Returns
  -------
  X_out, a np.ndarray with `len(X_out) == batch_size`.
  """
  num_samples = len(X_b)
  if num_samples > batch_size:
    raise ValueError("Cannot pad an array longer than `batch_size`")
  elif num_samples == batch_size:
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


def pad_batch(batch_size: int, X_b: np.ndarray, y_b: np.ndarray,
              w_b: np.ndarray, ids_b: np.ndarray) -> Batch:
  """Pads batch to have size precisely batch_size elements.

  Given arrays of features `X_b`, labels `y_b`, weights `w_b`, and
  identifiers `ids_b` all with length less than or equal to
  batch-size, pads them to `batch_size` length. It does this by
  repeating the original entries in tiled fashion. Note that `X_b,
  y_b, w_b, ids_b` must all have the same length.

  Parameters
  ----------
  batch_size: int
    The number of datapoints in a batch
  X_b: np.ndarray
    Must be such that `len(X_b) <= batch_size`
  y_b: np.ndarray
    Must be such that `len(y_b) <= batch_size`
  w_b: np.ndarray
    Must be such that `len(w_b) <= batch_size`
  ids_b: np.ndarray
    Must be such that `len(ids_b) <= batch_size`

  Returns
  -------
  (X_out, y_out, w_out, ids_out), all np.ndarray with length `batch_size`.
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
  elif len(y_b.shape) < 2:
    y_out = np.zeros(batch_size, dtype=y_b.dtype)
  else:
    y_out = np.zeros((batch_size,) + y_b.shape[1:], dtype=y_b.dtype)

  if w_b is None:
    w_out = None
  elif len(w_b.shape) < 2:
    w_out = np.zeros(batch_size, dtype=w_b.dtype)
  else:
    w_out = np.zeros((batch_size,) + w_b.shape[1:], dtype=w_b.dtype)

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
  """Abstract base class for datasets defined by X, y, w elements.

  `Dataset` objects are used to store representations of a dataset as
  used in a machine learning task. Datasets contain features `X`,
  labels `y`, weights `w` and identifiers `ids`. Different subclasses
  of `Dataset` may choose to hold `X, y, w, ids` in memory or on disk.

  The `Dataset` class attempts to provide for strong interoperability
  with other machine learning representations for datasets.
  Interconversion methods allow for `Dataset` objects to be converted
  to and from numpy arrays, pandas dataframes, tensorflow datasets,
  and pytorch datasets (only to and not from for pytorch at present).

  Note that you can never instantiate a `Dataset` object directly.
  Instead you will need to instantiate one of the concrete subclasses.
  """

  def __init__(self) -> None:
    raise NotImplementedError()

  def __len__(self) -> int:
    """
    Get the number of elements in the dataset.
    """
    raise NotImplementedError()

  def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
    """Get the shape of the dataset.

    Returns four tuples, giving the shape of the X, y, w, and ids
    arrays.
    """
    raise NotImplementedError()

  def get_task_names(self) -> np.ndarray:
    """Get the names of the tasks associated with this dataset."""
    raise NotImplementedError()

  @property
  def X(self) -> np.ndarray:
    """Get the X vector for this dataset as a single numpy array.

    Returns
    -------
    Numpy array of features `X`.

    Note
    ----
    If data is stored on disk, accesing this field may involve loading
    data from disk and could potentially be slow. Using
    `iterbatches()` or `itersamples()` may be more efficient for
    larger datasets.
    """
    raise NotImplementedError()

  @property
  def y(self) -> np.ndarray:
    """Get the y vector for this dataset as a single numpy array.

    Returns
    -------
    Numpy array of labels `y`.

    Note
    ----
    If data is stored on disk, accesing this field may involve loading
    data from disk and could potentially be slow. Using
    `iterbatches()` or `itersamples()` may be more efficient for
    larger datasets.
    """
    raise NotImplementedError()

  @property
  def ids(self) -> np.ndarray:
    """Get the ids vector for this dataset as a single numpy array.

    Returns
    -------
    Numpy array of identifiers `ids`.

    Note
    ----
    If data is stored on disk, accesing this field may involve loading
    data from disk and could potentially be slow. Using
    `iterbatches()` or `itersamples()` may be more efficient for
    larger datasets.
    """

    raise NotImplementedError()

  @property
  def w(self) -> np.ndarray:
    """Get the weight vector for this dataset as a single numpy array.

    Returns
    -------
    Numpy array of weights `w`.

    Note
    ----
    If data is stored on disk, accesing this field may involve loading
    data from disk and could potentially be slow. Using
    `iterbatches()` or `itersamples()` may be more efficient for
    larger datasets.
    """
    raise NotImplementedError()

  def __repr__(self) -> str:
    """Convert self to REPL print representation."""
    threshold = dc.utils.get_print_threshold()
    task_str = np.array2string(
        np.array(self.get_task_names()), threshold=threshold)
    if self.__len__() < dc.utils.get_max_print_size():
      id_str = np.array2string(self.ids, threshold=threshold)
      return "<%s X.shape: %s, y.shape: %s, w.shape: %s, ids: %s, task_names: %s>" % (
          self.__class__.__name__, str(self.X.shape), str(self.y.shape),
          str(self.w.shape), id_str, task_str)
    else:
      return "<%s X.shape: %s, y.shape: %s, w.shape: %s, task_names: %s>" % (
          self.__class__.__name__, str(self.X.shape), str(self.y.shape),
          str(self.w.shape), task_str)

  def __str__(self) -> str:
    """Convert self to str representation."""
    return self.__repr__()

  def iterbatches(self,
                  batch_size: Optional[int] = None,
                  epochs: int = 1,
                  deterministic: bool = False,
                  pad_batches: bool = False) -> Iterator[Batch]:
    """Get an object that iterates over minibatches from the dataset.

    Each minibatch is returned as a tuple of four numpy arrays: `(X,
    y, w, ids)`.

    Parameters
    ----------
    batch_size: int, optional
      Number of elements in each batch
    epochs: int, optional
      Number of epochs to walk over dataset
    deterministic: bool, optional
      If True, follow deterministic order.
    pad_batches: bool, optional
      If True, pad each batch to `batch_size`.

    Returns
    -------
    Generator which yields tuples of four numpy arrays `(X, y, w, ids)`
    """
    raise NotImplementedError()

  def itersamples(self) -> Iterator[Batch]:
    """Get an object that iterates over the samples in the dataset.

    Example:

    >>> dataset = NumpyDataset(np.ones((2,2)))
    >>> for x, y, w, id in dataset.itersamples():
    ...   print(x.tolist(), y.tolist(), w.tolist(), id)
    [1.0, 1.0] [0.0] [0.0] 0
    [1.0, 1.0] [0.0] [0.0] 1
    """
    raise NotImplementedError()

  def transform(self, transformer: "dc.trans.Transformer", **args) -> "Dataset":
    """Construct a new dataset by applying a transformation to every sample in this dataset.

    The argument is a function that can be called as follows:

    >> newx, newy, neww = fn(x, y, w)

    It might be called only once with the whole dataset, or multiple
    times with different subsets of the data.  Each time it is called,
    it should transform the samples and return the transformed data.

    Parameters
    ----------
    transformer: Transformer
      the transformation to apply to each sample in the dataset

    Returns
    -------
    a newly constructed Dataset object
    """
    raise NotImplementedError()

  def get_statistics(self, X_stats: bool = True,
                     y_stats: bool = True) -> Tuple[float, ...]:
    """Compute and return statistics of this dataset.

    Uses `self.itersamples()` to compute means and standard deviations
    of the dataset. Can compute on large datasets that don't fit in
    memory.

    Parameters
    ----------
    X_stats: bool, optional
      If True, compute feature-level mean and standard deviations.
    y_stats: bool, optional
      If True, compute label-level mean and standard deviations.

    Returns
    -------
    If `X_stats == True`, returns `(X_means, X_stds)`. If `y_stats == True`,
    returns `(y_means, y_stds)`. If both are true, returns
    `(X_means, X_stds, y_means, y_stds)`.
    """
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
      return tuple()

  def make_tf_dataset(self,
                      batch_size: int = 100,
                      epochs: int = 1,
                      deterministic: bool = False,
                      pad_batches: bool = False):
    """Create a tf.data.Dataset that iterates over the data in this Dataset.

    Each value returned by the Dataset's iterator is a tuple of (X, y,
    w) for one batch.

    Parameters
    ----------
    batch_size: int
      the number of samples to include in each batch
    epochs: int
      the number of times to iterate over the Dataset
    deterministic: bool
      if True, the data is produced in order.  If False, a different
      random permutation of the data is used for each epoch.
    pad_batches: bool
      if True, batches are padded as necessary to make the size of
      each batch exactly equal batch_size.

    Returns
    -------
    tf.Dataset that iterates over the same data.
    """
    # Retrieve the first sample so we can determine the dtypes.

    import tensorflow as tf
    X, y, w, ids = next(self.itersamples())
    dtypes = (tf.as_dtype(X.dtype), tf.as_dtype(y.dtype), tf.as_dtype(w.dtype))
    shapes = (tf.TensorShape([None] + list(X.shape)),
              tf.TensorShape([None] + list(y.shape)),
              tf.TensorShape([None] + list(w.shape)))

    # Create a Tensorflow Dataset.

    def gen_data():
      for X, y, w, ids in self.iterbatches(batch_size, epochs, deterministic,
                                           pad_batches):
        yield (X, y, w)

    return tf.data.Dataset.from_generator(gen_data, dtypes, shapes)

  def make_pytorch_dataset(self, epochs: int = 1, deterministic: bool = False):
    """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

    Each value returned by the Dataset's iterator is a tuple of (X, y,
    w, id) for one sample.

    Parameters
    ----------
    epochs: int
      the number of times to iterate over the Dataset
    deterministic: bool
      if True, the data is produced in order.  If False, a different
      random permutation of the data is used for each epoch.

    Returns
    -------
    `torch.utils.data.IterableDataset` that iterates over the data in
    this dataset.
    """
    raise NotImplementedError()

  def to_dataframe(self) -> pd.DataFrame:
    """Construct a pandas DataFrame containing the data from this Dataset.

    Returns
    -------
    pandas dataframe. If there is only a single feature per datapoint,
    will have column "X" else will have columns "X1,X2,..." for
    features.  If there is only a single label per datapoint, will
    have column "y" else will have columns "y1,y2,..." for labels. If
    there is only a single weight per datapoint will have column "w"
    else will have columns "w1,w2,...". Will have column "ids" for
    identifiers.
    """
    X = self.X
    y = self.y
    w = self.w
    ids = self.ids
    if len(X.shape) == 1 or X.shape[1] == 1:
      columns = ['X']
    else:
      columns = [f'X{i+1}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    if len(y.shape) == 1 or y.shape[1] == 1:
      columns = ['y']
    else:
      columns = [f'y{i+1}' for i in range(y.shape[1])]
    y_df = pd.DataFrame(y, columns=columns)
    if len(w.shape) == 1 or w.shape[1] == 1:
      columns = ['w']
    else:
      columns = [f'w{i+1}' for i in range(w.shape[1])]
    w_df = pd.DataFrame(w, columns=columns)
    ids_df = pd.DataFrame(ids, columns=['ids'])
    return pd.concat([X_df, y_df, w_df, ids_df], axis=1, sort=False)

  @staticmethod
  def from_dataframe(df: pd.DataFrame,
                     X: Optional[OneOrMany[str]] = None,
                     y: Optional[OneOrMany[str]] = None,
                     w: Optional[OneOrMany[str]] = None,
                     ids: Optional[str] = None):
    """Construct a Dataset from the contents of a pandas DataFrame.

    Parameters
    ----------
    df: DataFrame
      the pandas DataFrame
    X: string or list of strings
      the name of the column or columns containing the X array.  If
      this is None, it will look for default column names that match
      those produced by to_dataframe().
    y: string or list of strings
      the name of the column or columns containing the y array.  If
      this is None, it will look for default column names that match
      those produced by to_dataframe().
    w: string or list of strings
      the name of the column or columns containing the w array.  If
      this is None, it will look for default column names that match
      those produced by to_dataframe().
    ids: string
      the name of the column containing the ids.  If this is None, it
      will look for default column names that match those produced by
      to_dataframe().
    """
    # Find the X values.

    if X is not None:
      X_val = df[X]
    elif 'X' in df.columns:
      X_val = df['X']
    else:
      columns = []
      i = 1
      while f'X{i}' in df.columns:
        columns.append(f'X{i}')
        i += 1
      X_val = df[columns]
    if len(X_val.shape) == 1:
      X_val = np.expand_dims(X_val, 1)

    # Find the y values.

    if y is not None:
      y_val = df[y]
    elif 'y' in df.columns:
      y_val = df['y']
    else:
      columns = []
      i = 1
      while f'y{i}' in df.columns:
        columns.append(f'y{i}')
        i += 1
      y_val = df[columns]
    if len(y_val.shape) == 1:
      y_val = np.expand_dims(y_val, 1)

    # Find the w values.

    if w is not None:
      w_val = df[w]
    elif 'w' in df.columns:
      w_val = df['w']
    else:
      columns = []
      i = 1
      while f'w{i}' in df.columns:
        columns.append(f'w{i}')
        i += 1
      w_val = df[columns]
    if len(w_val.shape) == 1:
      w_val = np.expand_dims(w_val, 1)

    # Find the ids.

    if ids is not None:
      ids_val = df[ids]
    elif 'ids' in df.columns:
      ids_val = df['ids']
    else:
      ids_val = None
    return NumpyDataset(X_val, y_val, w_val, ids_val)


class NumpyDataset(Dataset):
  """A Dataset defined by in-memory numpy arrays.

  This subclass of `Dataset` stores arrays `X,y,w,ids` in memory as
  numpy arrays. This makes it very easy to construct `NumpyDataset`
  objects. For example

  >>> import numpy as np
  >>> dataset = NumpyDataset(X=np.random.rand(5, 3), y=np.random.rand(5,), ids=np.arange(5))
  """

  def __init__(self,
               X: np.ndarray,
               y: Optional[np.ndarray] = None,
               w: Optional[np.ndarray] = None,
               ids: Optional[np.ndarray] = None,
               n_tasks: int = 1) -> None:
    """Initialize this object.

    Parameters
    ----------
    X: np.ndarray
      Input features. Of shape `(n_samples,...)`
    y: np.ndarray, optional
      Labels. Of shape `(n_samples, ...)`. Note that each label can
      have an arbitrary shape.
    w: np.ndarray, optional
      Weights. Should either be 1D of shape `(n_samples,)` or if
      there's more than one task, of shape `(n_samples, n_tasks)`.
    ids: np.ndarray, optional
      Identifiers. Of shape `(n_samples,)`
    n_tasks: int, optional
      Number of learning tasks.
    """
    n_samples = len(X)
    if n_samples > 0:
      if y is None:
        # Set labels to be zero, with zero weights
        y = np.zeros((n_samples, n_tasks), np.float32)
        w = np.zeros((n_samples, 1), np.float32)
    if ids is None:
      ids = np.arange(n_samples)
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    if not isinstance(y, np.ndarray):
      y = np.array(y)
    if w is None:
      if len(y.shape) == 1:
        w = np.ones(y.shape[0], np.float32)
      else:
        w = np.ones((y.shape[0], 1), np.float32)
    if not isinstance(w, np.ndarray):
      w = np.array(w)
    self._X = X
    self._y = y
    self._w = w
    self._ids = np.array(ids, dtype=object)

  def __len__(self) -> int:
    """
    Get the number of elements in the dataset.
    """
    return len(self._y)

  def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
    """Get the shape of the dataset.

    Returns four tuples, giving the shape of the X, y, w, and ids
    arrays.
    """
    return self._X.shape, self._y.shape, self._w.shape, self._ids.shape

  def get_task_names(self) -> np.ndarray:
    """Get the names of the tasks associated with this dataset."""
    if len(self._y.shape) < 2:
      return np.array([0])
    return np.arange(self._y.shape[1])

  @property
  def X(self) -> np.ndarray:
    """Get the X vector for this dataset as a single numpy array."""
    return self._X

  @property
  def y(self) -> np.ndarray:
    """Get the y vector for this dataset as a single numpy array."""
    return self._y

  @property
  def ids(self) -> np.ndarray:
    """Get the ids vector for this dataset as a single numpy array."""
    return self._ids

  @property
  def w(self) -> np.ndarray:
    """Get the weight vector for this dataset as a single numpy array."""
    return self._w

  def iterbatches(self,
                  batch_size: Optional[int] = None,
                  epochs: int = 1,
                  deterministic: bool = False,
                  pad_batches: bool = False) -> Iterator[Batch]:
    """Get an object that iterates over minibatches from the dataset.

    Each minibatch is returned as a tuple of four numpy arrays: (X, y,
    w, ids).

    Parameters
    ----------
    batch_size: int, optional
      Number of elements in each batch
    epochs: int, optional
      Number of epochs to walk over dataset
    deterministic: bool, optional
      If True, follow deterministic order.
    pad_batches: bool, optional
      If True, pad each batch to `batch_size`.

    Returns
    -------
    Generator which yields tuples of four numpy arrays `(X, y, w, ids)`
    """

    def iterate(dataset: NumpyDataset, batch_size: Optional[int], epochs: int,
                deterministic: bool, pad_batches: bool):
      n_samples = dataset._X.shape[0]
      if deterministic:
        sample_perm = np.arange(n_samples)
      if batch_size is None:
        batch_size = n_samples
      for epoch in range(epochs):
        if not deterministic:
          sample_perm = np.random.permutation(n_samples)
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

    return iterate(self, batch_size, epochs, deterministic, pad_batches)

  def itersamples(self) -> Iterator[Batch]:
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

  def transform(self, transformer: "dc.trans.Transformer",
                **args) -> "NumpyDataset":
    """Construct a new dataset by applying a transformation to every sample in this dataset.

    The argument is a function that can be called as follows:

    >> newx, newy, neww = fn(x, y, w)

    It might be called only once with the whole dataset, or multiple
    times with different subsets of the data.  Each time it is called,
    it should transform the samples and return the transformed data.

    Parameters
    ----------
    transformer: Transformer
      the transformation to apply to each sample in the dataset

    Returns
    -------
    a newly constructed Dataset object
    """
    newx, newy, neww, newids = transformer.transform_array(
        self._X, self._y, self._w, self._ids)
    return NumpyDataset(newx, newy, neww, newids)

  def select(self, indices: Sequence[int],
             select_dir: str = None) -> "NumpyDataset":
    """Creates a new dataset from a selection of indices from self.

    Parameters
    ----------
    indices: list
      List of indices to select.
    select_dir: string
      Used to provide same API as `DiskDataset`. Ignored since
      `NumpyDataset` is purely in-memory.
    """
    X = self.X[indices]
    y = self.y[indices]
    w = self.w[indices]
    ids = self.ids[indices]
    return NumpyDataset(X, y, w, ids)

  def make_pytorch_dataset(self, epochs: int = 1, deterministic: bool = False):
    """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

    Each value returned by the Dataset's iterator is a tuple of (X, y, w, id) for
    one sample.

    Parameters
    ----------
    epochs: int
      the number of times to iterate over the Dataset
    deterministic: bool
      if True, the data is produced in order.  If False, a different random
      permutation of the data is used for each epoch.
    """
    import torch

    def iterate():
      n_samples = self._X.shape[0]
      worker_info = torch.utils.data.get_worker_info()
      if worker_info is None:
        first_sample = 0
        last_sample = n_samples
      else:
        first_sample = worker_info.id * n_samples // worker_info.num_workers
        last_sample = (
            worker_info.id + 1) * n_samples // worker_info.num_workers
      for epoch in range(epochs):
        if deterministic:
          order = first_sample + np.arange(last_sample - first_sample)
        else:
          order = first_sample + np.random.permutation(last_sample -
                                                       first_sample)
        for i in order:
          yield (self._X[i], self._y[i], self._w[i], self._ids[i])

    class TorchDataset(torch.utils.data.IterableDataset):  # type: ignore

      def __iter__(self):
        return iterate()

    return TorchDataset()

  @staticmethod
  def from_DiskDataset(ds: "DiskDataset") -> "NumpyDataset":
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
  def to_json(self, fname: str) -> None:
    d = {
        'X': self.X.tolist(),
        'y': self.y.tolist(),
        'w': self.w.tolist(),
        'ids': self.ids.tolist()
    }
    with open(fname, 'w') as fout:
      json.dump(d, fout)

  @staticmethod
  def from_json(fname: str) -> "NumpyDataset":
    with open(fname) as fin:
      d = json.load(fin)
      return NumpyDataset(d['X'], d['y'], d['w'], d['ids'])

  @staticmethod
  def merge(datasets: Sequence[Dataset]) -> "NumpyDataset":
    """
    Parameters
    ----------
    datasets: list of deepchem.data.Dataset
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

  def __init__(self, data_dir: str) -> None:
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    self.data_dir = data_dir

    logger.info("Loading dataset from disk.")
    self.tasks, self.metadata_df = self.load_metadata()
    self._cached_shards: Optional[List] = None
    self._memory_cache_size = 20 * (1 << 20)  # 20 MB
    self._cache_used = 0

  @staticmethod
  def create_dataset(shard_generator: Iterable[Batch],
                     data_dir: Optional[str] = None,
                     tasks: Optional[Sequence] = []) -> "DiskDataset":
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

    Returns
    -------
    A `DiskDataset` constructed from the given data
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
    logger.info("TIMING: dataset construction took %0.3f s" % (time2 - time1))
    return DiskDataset(data_dir)

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
  def _construct_metadata(metadata_entries: List) -> pd.DataFrame:
    """Construct a dataframe containing metadata.

    metadata_entries should have elements returned by write_data_to_disk
    above.
    """
    columns = ('ids', 'X', 'y', 'w')
    metadata_df = pd.DataFrame(metadata_entries, columns=columns)
    return metadata_df

  @staticmethod
  def write_data_to_disk(
      data_dir: str,
      basename: str,
      tasks: np.ndarray,
      X: Optional[np.ndarray] = None,
      y: Optional[np.ndarray] = None,
      w: Optional[np.ndarray] = None,
      ids: Optional[np.ndarray] = None) -> List[Optional[str]]:
    """Static helper method to write data to disk.

    This helper method is used to write a shard of data to disk.

    Parameters
    ----------
    data_dir: str
      Data directory to write shard to
    basename: str
      Basename for the shard in question.
    tasks: np.ndarray
      The names of the tasks in question.
    X: Optional[np.ndarray]
      The features array 
    y: Optional[np.ndarray]
      The labels array 
    w: Optional[np.ndarray]
      The weights array 
    ids: Optional[np.ndarray]
      The identifiers array 

    Returns
    -------
    List with values `[out_ids, out_X, out_y, out_w]` with filenames of locations to disk which these respective arrays were written.
    """
    if X is not None:
      out_X: Optional[str] = "%s-X.npy" % basename
      save_to_disk(X, os.path.join(data_dir, out_X))  # type: ignore
    else:
      out_X = None

    if y is not None:
      out_y: Optional[str] = "%s-y.npy" % basename
      save_to_disk(y, os.path.join(data_dir, out_y))  # type: ignore
    else:
      out_y = None

    if w is not None:
      out_w: Optional[str] = "%s-w.npy" % basename
      save_to_disk(w, os.path.join(data_dir, out_w))  # type: ignore
    else:
      out_w = None

    if ids is not None:
      out_ids: Optional[str] = "%s-ids.npy" % basename
      save_to_disk(ids, os.path.join(data_dir, out_ids))  # type: ignore
    else:
      out_ids = None

    # note that this corresponds to the _construct_metadata column order
    return [out_ids, out_X, out_y, out_w]

  def save_to_disk(self) -> None:
    """Save dataset to disk."""
    save_metadata(self.tasks, self.metadata_df, self.data_dir)
    self._cached_shards = None

  def move(self, new_data_dir: str) -> None:
    """Moves dataset to new directory."""
    if os.path.isdir(new_data_dir):
      shutil.rmtree(new_data_dir)
    shutil.move(self.data_dir, new_data_dir)
    self.data_dir = new_data_dir

  def get_task_names(self) -> np.ndarray:
    """
    Gets learning tasks associated with this dataset.
    """
    return self.tasks

  def reshard(self, shard_size: int) -> None:
    """Reshards data to have specified shard size."""
    # Create temp directory to store resharded version
    reshard_dir = tempfile.mkdtemp()

    n_shards = self.get_number_shards()

    # Write data in new shards
    def generator():
      tasks = self.get_task_names()
      X_next = np.zeros((0,) + self.get_data_shape())
      y_next = np.zeros((0,) + (len(tasks),))
      w_next = np.zeros((0,) + (len(tasks),))
      ids_next = np.zeros((0,), dtype=object)
      for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
        logger.info("Resharding shard %d/%d" % (shard_num, n_shards))
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
    # Note that this resets the cache internally
    self.save_to_disk()

  def get_data_shape(self) -> Shape:
    """
    Gets array shape of datapoints in this dataset.
    """
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_X = load_from_disk(
        os.path.join(self.data_dir,
                     next(self.metadata_df.iterrows())[1]['X']))
    return np.shape(sample_X)[1:]

  def get_shard_size(self) -> int:
    """Gets size of shards on disk."""
    if not len(self.metadata_df):
      raise ValueError("No data in dataset.")
    sample_y = load_from_disk(
        os.path.join(self.data_dir,
                     next(self.metadata_df.iterrows())[1]['y']))
    return len(sample_y)

  def _get_metadata_filename(self) -> Tuple[str, str]:
    """
    Get standard location for metadata file.
    """
    metadata_filename = os.path.join(self.data_dir, "metadata.csv.gzip")
    tasks_filename = os.path.join(self.data_dir, "tasks.json")
    return tasks_filename, metadata_filename

  def get_number_shards(self) -> int:
    """
    Returns the number of shards for this dataset.
    """
    return self.metadata_df.shape[0]

  def itershards(self) -> Iterator[Batch]:
    """
    Return an object that iterates over all shards in dataset.

    Datasets are stored in sharded fashion on disk. Each call to next() for the
    generator defined by this function returns the data from a particular shard.
    The order of shards returned is guaranteed to remain fixed.
    """
    return (self.get_shard(i) for i in range(self.get_number_shards()))

  def iterbatches(self,
                  batch_size: Optional[int] = None,
                  epochs: int = 1,
                  deterministic: bool = False,
                  pad_batches: bool = False) -> Iterator[Batch]:
    """ Get an object that iterates over minibatches from the dataset.

    It is guaranteed that the number of batches returned is
    `math.ceil(len(dataset)/batch_size)`. Each minibatch is returned as
    a tuple of four numpy arrays: `(X, y, w, ids)`.

    Parameters
    ----------
    batch_size: int
      Number of elements in a batch. If None, then it yields batches
      with size equal to the size of each individual shard.
    epoch: int
      Number of epochs to walk over dataset
    deterministic: bool
      Whether or not we should should shuffle each shard before
      generating the batches.  Note that this is only local in the
      sense that it does not ever mix between different shards.
    pad_batches: bool
      Whether or not we should pad the last batch, globally, such that
      it has exactly batch_size elements.
    """
    shard_indices = list(range(self.get_number_shards()))
    return self._iterbatches_from_shards(shard_indices, batch_size, epochs,
                                         deterministic, pad_batches)

  def _iterbatches_from_shards(self,
                               shard_indices: Sequence[int],
                               batch_size: Optional[int] = None,
                               epochs: int = 1,
                               deterministic: bool = False,
                               pad_batches: bool = False) -> Iterator[Batch]:
    """Get an object that iterates over batches from a restricted set of shards."""

    def iterate(dataset: DiskDataset, batch_size: Optional[int], epochs: int):
      num_shards = len(shard_indices)
      if deterministic:
        shard_perm = np.arange(num_shards)

      # (ytz): Depending on the application, thread-based pools may be faster
      # than process based pools, since process based pools need to pickle/serialize
      # objects as an extra overhead. Also, as hideously as un-thread safe this looks,
      # we're actually protected by the GIL.
      pool = multiprocessing.dummy.Pool(
          1)  # mp.dummy aliases ThreadPool to Pool

      if batch_size is None:
        num_global_batches = num_shards
      else:
        num_global_batches = math.ceil(dataset.get_shape()[0][0] / batch_size)

      for epoch in range(epochs):
        if not deterministic:
          shard_perm = np.random.permutation(num_shards)
        next_shard = pool.apply_async(dataset.get_shard,
                                      (shard_indices[shard_perm[0]],))
        cur_global_batch = 0
        cur_shard = 0
        carry = None

        while cur_global_batch < num_global_batches:

          X, y, w, ids = next_shard.get()
          if cur_shard < num_shards - 1:
            next_shard = pool.apply_async(
                dataset.get_shard, (shard_indices[shard_perm[cur_shard + 1]],))
          elif epoch == epochs - 1:
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

    return iterate(self, batch_size, epochs)

  def itersamples(self) -> Iterator[Batch]:
    """Get an object that iterates over the samples in the dataset.

    Example:

    >>> dataset = DiskDataset.from_numpy(np.ones((2,2)), np.ones((2,1)))
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

  def transform(self,
                transformer: "dc.trans.Transformer",
                parallel=False,
                **args) -> "DiskDataset":
    """Construct a new dataset by applying a transformation to every sample in this dataset.

    The argument is a function that can be called as follows:

    >> newx, newy, neww = fn(x, y, w)

    It might be called only once with the whole dataset, or multiple times
    with different subsets of the data.  Each time it is called, it should
    transform the samples and return the transformed data.

    Parameters
    ----------
    transformer: Transformer
      the transformation to apply to each sample in the dataset
    out_dir: string
      The directory to save the new dataset in.  If this is omitted, a
      temporary directory is created automatically
    parallel: bool
      if True, use multiple processes to transform the dataset in parallel

    Returns
    -------
    a newly constructed Dataset object
    """
    if 'out_dir' in args and args['out_dir'] is not None:
      out_dir = args['out_dir']
    else:
      out_dir = tempfile.mkdtemp()
    tasks = self.get_task_names()
    n_shards = self.get_number_shards()

    time1 = time.time()
    if parallel:
      results = []
      pool = multiprocessing.Pool()
      for i in range(self.get_number_shards()):
        row = self.metadata_df.iloc[i]
        X_file = os.path.join(self.data_dir, row['X'])
        if row['y'] is not None:
          y_file: Optional[str] = os.path.join(self.data_dir, row['y'])
        else:
          y_file = None
        if row['w'] is not None:
          w_file: Optional[str] = os.path.join(self.data_dir, row['w'])
        else:
          w_file = None
        ids_file = os.path.join(self.data_dir, row['ids'])
        results.append(
            pool.apply_async(DiskDataset._transform_shard,
                             (transformer, i, X_file, y_file, w_file, ids_file,
                              out_dir, tasks)))
      pool.close()
      metadata_rows = [r.get() for r in results]
      metadata_df = DiskDataset._construct_metadata(metadata_rows)
      save_metadata(tasks, metadata_df, out_dir)
      dataset = DiskDataset(out_dir)
    else:

      def generator():
        for shard_num, row in self.metadata_df.iterrows():
          logger.info("Transforming shard %d/%d" % (shard_num, n_shards))
          X, y, w, ids = self.get_shard(shard_num)
          newx, newy, neww, newids = transformer.transform_array(X, y, w, ids)
          yield (newx, newy, neww, newids)

      dataset = DiskDataset.create_dataset(
          generator(), data_dir=out_dir, tasks=tasks)
    time2 = time.time()
    logger.info("TIMING: transforming took %0.3f s" % (time2 - time1))
    return dataset

  @staticmethod
  def _transform_shard(transformer: "dc.trans.Transformer", shard_num: int,
                       X_file: str, y_file: str, w_file: str, ids_file: str,
                       out_dir: str, tasks: np.ndarray):
    """This is called by transform() to transform a single shard."""
    X = None if X_file is None else np.array(load_from_disk(X_file))
    y = None if y_file is None else np.array(load_from_disk(y_file))
    w = None if w_file is None else np.array(load_from_disk(w_file))
    ids = np.array(load_from_disk(ids_file))
    X, y, w, ids = transformer.transform_array(X, y, w, ids)
    basename = "shard-%d" % shard_num
    return DiskDataset.write_data_to_disk(out_dir, basename, tasks, X, y, w,
                                          ids)

  def make_pytorch_dataset(self, epochs: int = 1, deterministic: bool = False):
    """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

    Each value returned by the Dataset's iterator is a tuple of (X, y, w, id) for
    one sample.

    Parameters
    ----------
    epochs: int
      the number of times to iterate over the Dataset
    deterministic: bool
      if True, the data is produced in order.  If False, a different random
      permutation of the data is used for each epoch.
    """
    import torch

    def iterate():
      worker_info = torch.utils.data.get_worker_info()
      n_shards = self.get_number_shards()
      if worker_info is None:
        first_shard = 0
        last_shard = n_shards
      else:
        first_shard = worker_info.id * n_shards // worker_info.num_workers
        last_shard = (worker_info.id + 1) * n_shards // worker_info.num_workers
      if first_shard == last_shard:
        return
      shard_indices = list(range(first_shard, last_shard))
      for epoch in range(epochs):
        for X, y, w, ids in self._iterbatches_from_shards(
            shard_indices, deterministic=deterministic):
          for i in range(X.shape[0]):
            yield (X[i], y[i], w[i], ids[i])

    class TorchDataset(torch.utils.data.IterableDataset):  # type: ignore

      def __iter__(self):
        return iterate()

    return TorchDataset()

  @staticmethod
  def from_numpy(X: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 w: Optional[np.ndarray] = None,
                 ids: Optional[np.ndarray] = None,
                 tasks: Optional[Sequence] = None,
                 data_dir: Optional[str] = None) -> "DiskDataset":
    """Creates a DiskDataset object from specified Numpy arrays.

    Parameters
    ----------
    X: np.ndarray
      Feature array
    y: Optional[np.ndarray], optional (default None)
      labels array
    w: Optional[np.ndarray], optional (default None)
      weights array
    ids: Optional[np.ndarray], optional (default None)
      identifiers array
    tasks: Optional[Sequence], optional (default None)
      Tasks in this dataset
    data_dir: Optional[str], optional (default None)
      The directory to write this dataset to. If none is specified, will use
      a temporary dataset instead.

    Returns
    -------
    A `DiskDataset` constructed from the provided information.
    """
    n_samples = len(X)
    if ids is None:
      ids = np.arange(n_samples)

    if y is not None:
      if w is None:
        if len(y.shape) == 1:
          w = np.ones(y.shape[0], np.float32)
        else:
          w = np.ones((y.shape[0], 1), np.float32)

      if tasks is None:
        if len(y.shape) > 1:
          n_tasks = y.shape[1]
        else:
          n_tasks = 1
        tasks = np.arange(n_tasks)

    else:
      if w is not None:
        warnings.warn('y is None but w is not None. Setting w to None',
                      UserWarning)
        w = None

      if tasks is not None:
        warnings.warn('y is None but tasks is not None. Setting tasks to None',
                      UserWarning)
        tasks = None

    # raw_data = (X, y, w, ids)
    return DiskDataset.create_dataset(
        [(X, y, w, ids)], data_dir=data_dir, tasks=tasks)

  @staticmethod
  def merge(datasets: Iterable["DiskDataset"],
            merge_dir: Optional[str] = None) -> "DiskDataset":
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
        logger.info("Merging in dataset %d/%d" % (ind, len(datasets)))
        X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
        yield (X, y, w, ids)

    return DiskDataset.create_dataset(
        generator(), data_dir=merge_dir, tasks=tasks)

  def subset(self, shard_nums: Sequence[int],
             subset_dir: Optional[str] = None) -> "DiskDataset":
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

  def sparse_shuffle(self) -> None:
    """Shuffling that exploits data sparsity to shuffle large datasets.

    Only for 1-dimensional feature vectors (does not work for tensorial
    featurizations).
    """
    time1 = time.time()
    shard_size = self.get_shard_size()
    num_shards = self.get_number_shards()
    X_sparses: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    ids: List[np.ndarray] = []
    num_features = -1
    for i in range(num_shards):
      logger.info("Sparsifying shard %d/%d" % (i, num_shards))
      (X_s, y_s, w_s, ids_s) = self.get_shard(i)
      if num_features == -1:
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
      logger.info("Sparse shuffling shard %d/%d" % (i, num_shards))
      start, stop = i * shard_size, (i + 1) * shard_size
      (X_sparse_s, y_s, w_s, ids_s) = (X_sparse[start:stop], y[start:stop],
                                       w[start:stop], ids[start:stop])
      X_s = densify_features(X_sparse_s, num_features)
      self.set_shard(i, X_s, y_s, w_s, ids_s)
    time2 = time.time()
    logger.info("TIMING: sparse_shuffle took %0.3f s" % (time2 - time1))

  def complete_shuffle(self, data_dir: Optional[str] = None) -> "DiskDataset":
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
    DiskDataset
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

    Xs = np.concatenate(all_X)
    ys = np.concatenate(all_y)
    ws = np.concatenate(all_w)
    ids = np.concatenate(all_ids)

    perm = np.random.permutation(Xs.shape[0])
    Xs = Xs[perm]
    ys = ys[perm]
    ws = ws[perm]
    ids = ids[perm]

    return DiskDataset.from_numpy(Xs, ys, ws, ids, data_dir=data_dir)

  def shuffle_each_shard(self,
                         shard_basenames: Optional[List[str]] = None) -> None:
    """Shuffles elements within each shard of the datset.

    Parameters
    ----------
    shard_basenames: Optional[List[str]], optional (default None)
      The basenames for each shard. If this isn't specified, will assume the
       basenames of form "shard-i" used by `create_dataset` and
      `reshard`.
    """
    tasks = self.get_task_names()
    # Shuffle the arrays corresponding to each row in metadata_df
    n_rows = len(self.metadata_df.index)
    if shard_basenames is not None:
      if len(shard_basenames) != n_rows:
        raise ValueError(
            "shard_basenames must provide a basename for each shard in this DiskDataset."
        )
    else:
      shard_basenames = ["shard-%d" % shard_num for shard_num in range(n_rows)]
    for i, basename in zip(range(n_rows), shard_basenames):
      logger.info("Shuffling shard %d/%d" % (i, n_rows))
      X, y, w, ids = self.get_shard(i)
      n = X.shape[0]
      permutation = np.random.permutation(n)
      X, y, w, ids = (X[permutation], y[permutation], w[permutation],
                      ids[permutation])
      DiskDataset.write_data_to_disk(self.data_dir, basename, tasks, X, y, w,
                                     ids)
    # Reset cache
    self._cached_shards = None

  def shuffle_shards(self) -> None:
    """Shuffles the order of the shards for this dataset."""
    metadata_rows = self.metadata_df.values.tolist()
    random.shuffle(metadata_rows)
    self.metadata_df = DiskDataset._construct_metadata(metadata_rows)
    self.save_to_disk()

  def get_shard(self, i: int) -> Batch:
    """Retrieves data for the i-th shard from disk."""

    class Shard(object):

      def __init__(self, X, y, w, ids):
        self.X = X
        self.y = y
        self.w = w
        self.ids = ids

    # See if we have a cached copy of this shard.
    if self._cached_shards is None:
      self._cached_shards = [None] * self.get_number_shards()
      self._cache_used = 0
    if self._cached_shards[i] is not None:
      shard = self._cached_shards[i]
      return (shard.X, shard.y, shard.w, shard.ids)

    # We don't, so load it from disk.
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
        if len(y.shape) == 1:
          w = np.ones(y.shape[0], np.float32)
        else:
          w = np.ones((y.shape[0], 1), np.float32)
    else:
      w = None

    ids = np.array(
        load_from_disk(os.path.join(self.data_dir, row['ids'])), dtype=object)

    # Try to cache this shard for later use.  Since the normal usage pattern is
    # a series of passes through the whole dataset, there's no point doing
    # anything fancy.  It never makes sense to evict another shard from the
    # cache to make room for this one, because we'll probably want that other
    # shard again before the next time we want this one.  So just cache as many
    # as we can and then stop.

    shard = Shard(X, y, w, ids)
    shard_size = X.nbytes + ids.nbytes
    if y is not None:
      shard_size += y.nbytes
    if w is not None:
      shard_size += w.nbytes
    if self._cache_used + shard_size < self._memory_cache_size:
      self._cached_shards[i] = shard
      self._cache_used += shard_size
    return (shard.X, shard.y, shard.w, shard.ids)

  def get_shard_ids(self, i: int) -> np.ndarray:
    """Retrieves the list of IDs for the i-th shard from disk."""

    if self._cached_shards is not None and self._cached_shards[i] is not None:
      return self._cached_shards[i].ids
    row = self.metadata_df.iloc[i]
    return np.array(
        load_from_disk(os.path.join(self.data_dir, row['ids'])), dtype=object)

  def get_shard_y(self, i: int) -> np.ndarray:
    """Retrieves the labels for the i-th shard from disk.

    Parameters
    ----------
    i: int
      Shard index for shard to retrieve labels from
    """

    if self._cached_shards is not None and self._cached_shards[i] is not None:
      return self._cached_shards[i].y
    row = self.metadata_df.iloc[i]
    return np.array(
        load_from_disk(os.path.join(self.data_dir, row['y'])), dtype=object)

  def get_shard_w(self, i: int) -> np.ndarray:
    """Retrieves the weights for the i-th shard from disk.

    Parameters
    ----------
    i: int
      Shard index for shard to retrieve weights from
    """

    if self._cached_shards is not None and self._cached_shards[i] is not None:
      return self._cached_shards[i].w
    row = self.metadata_df.iloc[i]
    return np.array(
        load_from_disk(os.path.join(self.data_dir, row['w'])), dtype=object)

  def add_shard(self, X: np.ndarray, y: Optional[np.ndarray],
                w: Optional[np.ndarray], ids: Optional[np.ndarray]) -> None:
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

  def set_shard(self, shard_num: int, X: np.ndarray, y: Optional[np.ndarray],
                w: Optional[np.ndarray], ids: Optional[np.ndarray]) -> None:
    """Writes data shard to disk"""
    basename = "shard-%d" % shard_num
    tasks = self.get_task_names()
    DiskDataset.write_data_to_disk(self.data_dir, basename, tasks, X, y, w, ids)
    self._cached_shards = None

  def select(self, indices: Sequence[int],
             select_dir: str = None) -> "DiskDataset":
    """Creates a new dataset from a selection of indices from self.

    Parameters
    ----------
    indices: list
      List of indices to select.
    select_dir: string
      Path to new directory that the selected indices will be copied
      to.
    """
    if select_dir is not None:
      if not os.path.exists(select_dir):
        os.makedirs(select_dir)
    else:
      select_dir = tempfile.mkdtemp()
    # Handle edge case with empty indices
    if not len(indices):
      return DiskDataset.create_dataset([], data_dir=select_dir)
    indices = np.array(sorted(indices)).astype(int)
    tasks = self.get_task_names()

    n_shards = self.get_number_shards()

    def generator():
      count, indices_count = 0, 0
      for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
        logger.info("Selecting from shard %d/%d" % (shard_num, n_shards))
        shard_len = len(X)
        # Find indices which rest in this shard
        num_shard_elts = 0
        while indices[indices_count + num_shard_elts] < count + shard_len:
          num_shard_elts += 1
          if indices_count + num_shard_elts >= len(indices):
            break
        # Need to offset indices to fit within shard_size
        shard_inds = indices[indices_count:indices_count +
                             num_shard_elts] - count
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
        generator(), data_dir=select_dir, tasks=tasks)

  @property
  def ids(self) -> np.ndarray:
    """Get the ids vector for this dataset as a single numpy array."""
    if len(self) == 0:
      return np.array([])
    ids = []
    for i in range(self.get_number_shards()):
      ids.append(np.atleast_1d(np.squeeze(self.get_shard_ids(i))))
    return np.concatenate(ids)

  @property
  def X(self) -> np.ndarray:
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
  def y(self) -> np.ndarray:
    """Get the y vector for this dataset as a single numpy array."""
    if len(self) == 0:
      return np.array([])
    ys = []
    one_dimensional = False
    for i in range(self.get_number_shards()):
      y_b = self.get_shard_y(i)
      ys.append(y_b)
      if len(y_b.shape) == 1:
        one_dimensional = True
    if not one_dimensional:
      return np.vstack(ys)
    else:
      return np.concatenate(ys)

  @property
  def w(self) -> np.ndarray:
    """Get the weight vector for this dataset as a single numpy array."""
    ws = []
    one_dimensional = False
    for i in range(self.get_number_shards()):
      w_b = self.get_shard_w(i)
      ws.append(w_b)
      if len(w_b.shape) == 1:
        one_dimensional = True
    if not one_dimensional:
      return np.vstack(ws)
    else:
      return np.concatenate(ws)

  @property
  def memory_cache_size(self) -> int:
    """Get the size of the memory cache for this dataset, measured in bytes."""
    return self._memory_cache_size

  @memory_cache_size.setter
  def memory_cache_size(self, size: int) -> None:
    """Get the size of the memory cache for this dataset, measured in bytes."""
    self._memory_cache_size = size
    if self._cache_used > size:
      self._cached_shards = None

  def __len__(self) -> int:
    """
    Finds number of elements in dataset.
    """
    total = 0
    for _, row in self.metadata_df.iterrows():
      y = load_from_disk(os.path.join(self.data_dir, row['ids']))
      total += len(y)
    return total

  def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
    """Finds shape of dataset."""
    n_tasks = len(self.get_task_names())
    for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
      if shard_num == 0:
        X_shape = np.array(X.shape)
        if n_tasks > 0:
          y_shape = np.array(y.shape)
          w_shape = np.array(w.shape)
        else:
          y_shape = tuple()
          w_shape = tuple()
        ids_shape = np.array(ids.shape)
      else:
        X_shape[0] += np.array(X.shape)[0]
        if n_tasks > 0:
          y_shape[0] += np.array(y.shape)[0]
          w_shape[0] += np.array(w.shape)[0]
        ids_shape[0] += np.array(ids.shape)[0]
    return tuple(X_shape), tuple(y_shape), tuple(w_shape), tuple(ids_shape)

  def get_label_means(self) -> pd.DataFrame:
    """Return pandas series of label means."""
    return self.metadata_df["y_means"]

  def get_label_stds(self) -> pd.DataFrame:
    """Return pandas series of label stds."""
    return self.metadata_df["y_stds"]


class ImageDataset(Dataset):
  """A Dataset that loads data from image files on disk."""

  def __init__(self,
               X: Union[np.ndarray, List[str]],
               y: Optional[Union[np.ndarray, List[str]]],
               w: Optional[Sequence] = None,
               ids: Optional[Sequence] = None) -> None:
    """Create a dataset whose X and/or y array is defined by image files on disk.

    Parameters
    ----------
    X: ndarray or list of strings
      The dataset's input data.  This may be either a single NumPy
      array directly containing the data, or a list containing the
      paths to the image files
    y: ndarray or list of strings
      The dataset's labels.  This may be either a single NumPy array
      directly containing the data, or a list containing the paths to
      the image files
    w: ndarray, optional, (default, None)
      a 1D or 2D array containing the weights for each sample or
      sample/task pair
    ids: ndarray, optional (default None)
      the sample IDs
    """
    n_samples = len(X)
    if y is None:
      y = np.zeros((n_samples,))
    self._X_shape = self._find_array_shape(X)
    self._y_shape = self._find_array_shape(y)
    if w is None:
      if len(self._y_shape) == 0:
        # Case n_samples should be 1
        if n_samples != 1:
          raise ValueError("y can only be a scalar if n_samples == 1")
        w = np.ones_like(y)
      elif len(self._y_shape) == 1:
        w = np.ones(self._y_shape[0], np.float32)
      else:
        w = np.ones((self._y_shape[0], 1), np.float32)
    if ids is None:
      if not isinstance(X, np.ndarray):
        ids = X
      elif not isinstance(y, np.ndarray):
        ids = y
      else:
        ids = np.arange(n_samples)
    self._X = X
    self._y = y
    self._w: np.ndarray = w
    self._ids = np.array(ids, dtype=object)

  def _find_array_shape(self, array: Sequence) -> Shape:
    if isinstance(array, np.ndarray):
      return array.shape
    image_shape = dc.data.ImageLoader.load_img([array[0]]).shape[1:]
    return np.concatenate([[len(array)], image_shape])

  def __len__(self) -> int:
    """
    Get the number of elements in the dataset.
    """
    return self._X_shape[0]

  def get_shape(self) -> Tuple[Shape, Shape, Shape, Shape]:
    """Get the shape of the dataset.

    Returns four tuples, giving the shape of the X, y, w, and ids
    arrays.
    """
    return self._X_shape, self._y_shape, self._w.shape, self._ids.shape

  def get_task_names(self) -> np.ndarray:
    """Get the names of the tasks associated with this dataset."""
    if len(self._y_shape) < 2:
      return np.array([0])
    return np.arange(self._y_shape[1])

  @property
  def X(self) -> np.ndarray:
    """Get the X vector for this dataset as a single numpy array."""
    if isinstance(self._X, np.ndarray):
      return self._X
    return dc.data.ImageLoader.load_img(self._X)

  @property
  def y(self) -> np.ndarray:
    """Get the y vector for this dataset as a single numpy array."""
    if isinstance(self._y, np.ndarray):
      return self._y
    return dc.data.ImageLoader.load_img(self._y)

  @property
  def ids(self) -> np.ndarray:
    """Get the ids vector for this dataset as a single numpy array."""
    return self._ids

  @property
  def w(self) -> np.ndarray:
    """Get the weight vector for this dataset as a single numpy array."""
    return self._w

  def iterbatches(self,
                  batch_size: Optional[int] = None,
                  epochs: int = 1,
                  deterministic: bool = False,
                  pad_batches: bool = False) -> Iterator[Batch]:
    """Get an object that iterates over minibatches from the dataset.

    Each minibatch is returned as a tuple of four numpy arrays: (X, y,
    w, ids).
    """

    def iterate(dataset, batch_size, epochs, deterministic, pad_batches):
      n_samples = dataset._X_shape[0]
      if deterministic:
        sample_perm = np.arange(n_samples)
      if batch_size is None:
        batch_size = n_samples
      for epoch in range(epochs):
        if not deterministic:
          sample_perm = np.random.permutation(n_samples)
        batch_idx = 0
        num_batches = np.math.ceil(n_samples / batch_size)
        while batch_idx < num_batches:
          start = batch_idx * batch_size
          end = min(n_samples, (batch_idx + 1) * batch_size)
          indices = range(start, end)
          perm_indices = sample_perm[indices]
          if isinstance(dataset._X, np.ndarray):
            X_batch = dataset._X[perm_indices]
          else:
            X_batch = dc.data.ImageLoader.load_img(
                [dataset._X[i] for i in perm_indices])
          if isinstance(dataset._y, np.ndarray):
            y_batch = dataset._y[perm_indices]
          else:
            y_batch = dc.data.ImageLoader.load_img(
                [dataset._y[i] for i in perm_indices])
          w_batch = dataset._w[perm_indices]
          ids_batch = dataset._ids[perm_indices]
          if pad_batches:
            (X_batch, y_batch, w_batch, ids_batch) = pad_batch(
                batch_size, X_batch, y_batch, w_batch, ids_batch)
          batch_idx += 1
          yield (X_batch, y_batch, w_batch, ids_batch)

    return iterate(self, batch_size, epochs, deterministic, pad_batches)

  def itersamples(self) -> Iterator[Batch]:
    """Get an object that iterates over the samples in the dataset.

    Example:

    >>> dataset = NumpyDataset(np.ones((2,2)))
    >>> for x, y, w, id in dataset.itersamples():
    ...   print(x.tolist(), y.tolist(), w.tolist(), id)
    [1.0, 1.0] [0.0] [0.0] 0
    [1.0, 1.0] [0.0] [0.0] 1
    """

    def get_image(array, index):
      if isinstance(array, np.ndarray):
        return array[index]
      return dc.data.ImageLoader.load_img([array[index]])[0]

    n_samples = self._X_shape[0]
    return ((get_image(self._X, i), get_image(self._y, i), self._w[i],
             self._ids[i]) for i in range(n_samples))

  def transform(self, transformer: "dc.trans.Transformer",
                **args) -> NumpyDataset:
    """Construct a new dataset by applying a transformation to every sample in this dataset.

    The argument is a function that can be called as follows:

    >> newx, newy, neww = fn(x, y, w)

    It might be called only once with the whole dataset, or multiple times with
    different subsets of the data.  Each time it is called, it should transform
    the samples and return the transformed data.

    Parameters
    ----------
    transformer: Transformer
      the transformation to apply to each sample in the dataset

    Returns
    -------
    a newly constructed Dataset object
    """
    newx, newy, neww, newids = transformer.transform_array(
        self.X, self.y, self.w, self.ids)
    return NumpyDataset(newx, newy, neww, newids)

  def select(self, indices: Sequence[int],
             select_dir: str = None) -> "ImageDataset":
    """Creates a new dataset from a selection of indices from self.

    Parameters
    ----------
    indices: list
      List of indices to select.
    select_dir: string
      Used to provide same API as `DiskDataset`. Ignored since
      `ImageDataset` is purely in-memory.
    """
    if isinstance(self._X, np.ndarray):
      X = self._X[indices]
    else:
      X = [self._X[i] for i in indices]
    if isinstance(self._y, np.ndarray):
      y = self._y[indices]
    else:
      y = [self._y[i] for i in indices]
    w = self._w[indices]
    ids = self._ids[indices]
    return ImageDataset(X, y, w, ids)

  def make_pytorch_dataset(self, epochs: int = 1, deterministic: bool = False):
    """Create a torch.utils.data.IterableDataset that iterates over the data in this Dataset.

    Each value returned by the Dataset's iterator is a tuple of (X, y,
    w, id) for one sample.

    Parameters
    ----------
    epochs: int
      the number of times to iterate over the Dataset
    deterministic: bool
      if True, the data is produced in order.  If False, a different
      random permutation of the data is used for each epoch.

    Returns
    -------
    `torch.utils.data.IterableDataset` iterating over the same data as
    this dataset.
    """
    import torch

    def get_image(array, index):
      if isinstance(array, np.ndarray):
        return array[index]
      return dc.data.ImageLoader.load_img([array[index]])[0]

    def iterate():
      n_samples = self._X_shape[0]
      worker_info = torch.utils.data.get_worker_info()
      if worker_info is None:
        first_sample = 0
        last_sample = n_samples
      else:
        first_sample = worker_info.id * n_samples // worker_info.num_workers
        last_sample = (
            worker_info.id + 1) * n_samples // worker_info.num_workers
      for epoch in range(epochs):
        if deterministic:
          order = first_sample + np.arange(last_sample - first_sample)
        else:
          order = first_sample + np.random.permutation(last_sample -
                                                       first_sample)
        for i in order:
          yield (get_image(self._X, i), get_image(self._y, i), self._w[i],
                 self._ids[i])

    class TorchDataset(torch.utils.data.IterableDataset):  # type: ignore

      def __iter__(self):
        return iterate()

    return TorchDataset()


class Databag(object):
  """A utility class to iterate through multiple datasets together.


  A `Databag` is useful when you have multiple datasets that you want
  to iterate in locksteps. This might be easiest to grasp with a
  simple code example.

  >>> ones_dataset = NumpyDataset(X=np.ones((5, 3)))
  >>> zeros_dataset = NumpyDataset(X=np.zeros((5, 3)))
  >>> databag = Databag({"ones": ones_dataset, "zeros": zeros_dataset})
  >>> for sample_dict in databag.iterbatches(batch_size=1):
  ...   print(sample_dict)
  {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
  {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
  {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
  {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}
  {'ones': array([[1., 1., 1.]]), 'zeros': array([[0., 0., 0.]])}

  Note how we get a batch at a time from each of the datasets in the
  `Databag`. This can be useful for training models that combine data
  from multiple `Dataset` objects at a time.
  """

  def __init__(self, datasets: Optional[Dict[Any, Dataset]] = None) -> None:
    """Initialize this `Databag`.

    Parameters
    ----------
    datasets: dict, optional
      A dictionary mapping keys to `Dataset` objects.
    """
    if datasets is None:
      self.datasets = dict()
    else:
      self.datasets = datasets

  def add_dataset(self, key: Any, dataset: Dataset) -> None:
    """Adds a dataset to this databag.

    Parameters
    ----------
    key: hashable value
      Key to be added
    dataset: `Dataset`
      The dataset that `key` should point to.
    """
    self.datasets[key] = dataset

  def iterbatches(self, **kwargs) -> Iterator[Dict[Any, Dataset]]:
    """Loop through all internal datasets in the same order.

    Parameters
    ----------
    batch_size: int
      Number of samples from each dataset to return
    epochs: int
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
