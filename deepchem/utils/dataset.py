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
  def __init__(self, data_dir, samples=None, feature_types=None):
    """
    Turns featurized dataframes into numpy files, writes them & metadata to disk.
    """
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    self.data_dir = data_dir

    if samples is not None and feature_types is not None:
      if not isinstance(feature_types, list):
        raise ValueError("feature_types must be a list or None.")

      write_dataset_single_partial = partial(
          write_dataset_single, data_dir=self.data_dir,
          feature_types=feature_types)

      metadata_rows = []
      # TODO(rbharath): Still a bit of information leakage.
      for df_file, df in zip(samples.dataset_files, samples.itersamples()):
        retval = write_dataset_single_partial((df_file, df))
        if retval is not None:
          metadata_rows.append(retval)

      # TODO(rbharath): FeaturizedSamples should not be responsible for
      # X-transform, X_sums, etc. Move that stuff over to Dataset.
      self.metadata_df = pd.DataFrame(
          metadata_rows,
          columns=('df_file', 'task_names', 'ids',
                   'X', 'X-transformed', 'y', 'y-transformed',
                   'w',
                   'X_sums', 'X_sum_squares', 'X_n',
                   'y_sums', 'y_sum_squares', 'y_n'))
      save_to_disk(
          self.metadata_df, self._get_metadata_filename())
      # input/output transforms not specified yet, so
      # self.transforms = (input_transforms, output_transforms) =>
      self.transforms = ([], [])
      save_to_disk(
          self.transforms, self._get_transforms_filename())
    else:
      if os.path.exists(self._get_metadata_filename()):
        self.metadata_df = load_from_disk(self._get_metadata_filename())
        self.transforms = load_from_disk(self._get_transforms_filename())
      else:
        raise ValueError("No metadata found.")

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

  def _get_transforms_filename(self):
    """
    Get standard location for stored transforms.
    """
    return os.path.join(self.data_dir, "transforms.joblib")

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

  def transform(self, input_transforms, output_transforms, parallel=False):
    """
    Transforms all internally stored data.

    Adds X-transform, y-transform columns to metadata.
    """
    (normalize_X, truncate_x, normalize_y, truncate_y, log_X, log_y) = (
        False, False, False, False, False, False)

    if "truncate" in input_transforms:
      truncate_x = True
    if "normalize" in input_transforms:
      normalize_X = True
    if "log" in input_transforms:
      log_X = True

    if "normalize" in output_transforms:
      normalize_y = True
    if "log" in output_transforms:
      log_y = True

    # Store input_transforms/output_transforms so the dataset remembers its state.

    X_means, X_stds, y_means, y_stds = self._transform(normalize_X, normalize_y,
                                                       truncate_x, truncate_y,
                                                       log_X, log_y,
                                                       parallel=parallel)
    nrow = self.metadata_df.shape[0]
    # TODO(rbharath): These lines are puzzling. Better way to avoid storage
    # duplication here?
    self.metadata_df['X_means'] = [X_means for _ in range(nrow)]
    self.metadata_df['X_stds'] = [X_stds for _ in range(nrow)]
    self.metadata_df['y_means'] = [y_means for _ in range(nrow)]
    self.metadata_df['y_stds'] = [y_stds for _ in range(nrow)]
    save_to_disk(
        self.metadata_df, self._get_metadata_filename())
    self.transforms = (input_transforms, output_transforms)
    save_to_disk(
        self.transforms, self._get_transforms_filename())

  def get_label_means(self):
    """Return pandas series of label means."""
    return self.metadata_df["y_means"]

  def get_label_stds(self):
    """Return pandas series of label stds."""
    return self.metadata_df["y_stds"]

  def get_input_transforms(self):
    """Returns stored input transforms."""
    (input_transforms, _) = self.transforms
    return input_transforms

  def get_output_transforms(self):
    """Returns stored output transforms."""
    (_, output_transforms) = self.transforms
    return output_transforms

  def _transform(self, normalize_X=True, normalize_y=True,
                 truncate_X=True, truncate_y=True,
                 log_X=False, log_y=False, parallel=False):
    """Helper to (parallel) transform all indexed data."""
    df = self.metadata_df
    trunc = 5.0
    X_means, X_stds, y_means, y_stds = compute_mean_and_std(df)
    indices = range(0, df.shape[0])
    transform_row_partial = partial(_transform_row, df=df, normalize_X=normalize_X,
                                    normalize_y=normalize_y, truncate_X=truncate_X,
                                    truncate_y=truncate_y, log_X=log_X,
                                    log_y=log_y, X_means=X_means, X_stds=X_stds,
                                    y_means=y_means, y_stds=y_stds, trunc=trunc)
    if parallel:
      pool = mp.Pool(int(mp.cpu_count()/4))
      pool.map(transform_row_partial, indices)
      pool.terminate()
    else:
      for index in indices:
        transform_row_partial(index)

    return X_means, X_stds, y_means, y_stds

def _transform_row(i, df, normalize_X, normalize_y, truncate_X, truncate_y,
                   log_X, log_y, X_means, X_stds, y_means, y_stds, trunc):
  """
  Transforms the data (X, y, w,...) in a single row.

  Writes X-transforme,d y-transformed to disk.
  """
  row = df.iloc[i]
  X = load_from_disk(row['X'])
  if normalize_X or log_X:
    if normalize_X:
      # Turns NaNs to zeros
      X = np.nan_to_num((X - X_means) / X_stds)
      if truncate_X:
        X[X > trunc] = trunc
        X[X < (-1.0*trunc)] = -1.0 * trunc
    if log_X:
      X = np.log(X)
  save_to_disk(X, row['X-transformed'])

  y = load_from_disk(row['y'])
  if normalize_y or log_y:
    if normalize_y:
      y = np.nan_to_num((y - y_means) / y_stds)
      if truncate_y:
        y[y > trunc] = trunc
        y[y < (-1.0*trunc)] = -1.0 * trunc
    if log_y:
      y = np.log(y)
  save_to_disk(y, row['y-transformed'])

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

def write_dataset_single(val, data_dir, feature_types):
  """Writes files for single row (X, y, w, X-transformed, ...) to disk."""
  (df_file, df) = val
  # TODO(rbharath): This is a hack. clean up.
  if not len(df):
    return None
  task_names = FeaturizedSamples.get_sorted_task_names(df)
  ids, X, y, w = _df_to_numpy(df, feature_types)
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
  save_to_disk(w, out_w)
  save_to_disk(ids, out_ids)
  # TODO(rbharath): Should X be saved to out_X_transformed as well? Since
  # itershards expects to loop over X-transformed? (Ditto for y/w)
  return([df_file, task_names, out_ids, out_X, out_X_transformed, out_y,
          out_y_transformed, out_w,
          X_sums, X_sum_squares, X_n,
          y_sums, y_sum_squares, y_n])

def _df_to_numpy(df, feature_types):
  """Transforms a featurized dataset df into standard set of numpy arrays"""
  if not set(feature_types).issubset(df.keys()):
    raise ValueError(
        "Featurized data does not support requested feature_types.")
  # perform common train/test split across all tasks
  n_samples = df.shape[0]
  sorted_tasks = FeaturizedSamples.get_sorted_task_names(df)
  n_tasks = len(sorted_tasks)
  y = df[sorted_tasks].values
  y = np.reshape(y, (n_samples, n_tasks))
  w = np.ones((n_samples, n_tasks))
  tensors = []
  for _, datapoint in df.iterrows():
    feature_list = []
    for feature_type in feature_types:
      feature_list.append(datapoint[feature_type])
    features = np.squeeze(np.concatenate(feature_list))
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
