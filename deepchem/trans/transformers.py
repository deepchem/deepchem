# coding=utf-8
"""
Contains an abstract base class that supports data transformations.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import scipy
import scipy.ndimage
import time
import deepchem as dc
import tensorflow as tf
from deepchem.data import NumpyDataset


def undo_transforms(y, transformers):
  """Undoes all transformations applied."""
  # Note that transformers have to be undone in reversed order
  for transformer in reversed(transformers):
    if transformer.transform_y:
      y = transformer.untransform(y)
  return y


def undo_grad_transforms(grad, tasks, transformers):
  for transformer in reversed(transformers):
    if transformer.transform_y:
      grad = transformer.untransform_grad(grad, tasks)
  return grad


def get_grad_statistics(dataset):
  """Computes and returns statistics of a dataset

  This function assumes that the first task of a dataset holds the energy for
  an input system, and that the remaining tasks holds the gradient for the
  system.
  """
  if len(dataset) == 0:
    return None, None, None, None
  y = dataset.y
  energy = y[:, 0]
  grad = y[:, 1:]
  for i in range(energy.size):
    grad[i] *= energy[i]
  ydely_means = np.sum(grad, axis=0) / len(energy)
  return grad, ydely_means


class Transformer(object):
  """
  Abstract base class for different ML models.
  """
  # Hack to allow for easy unpickling:
  # http://stefaanlippens.net/pickleproblem
  __module__ = os.path.splitext(os.path.basename(__file__))[0]

  def __init__(self,
               transform_X=False,
               transform_y=False,
               transform_w=False,
               dataset=None):
    """Initializes transformation based on dataset statistics."""
    self.dataset = dataset
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.transform_w = transform_w
    # One, but not both, transform_X or tranform_y is true
    assert transform_X or transform_y or transform_w
    # Use fact that bools add as ints in python
    assert (transform_X + transform_y + transform_w) == 1

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    raise NotImplementedError(
        "Each Transformer is responsible for its own transform_array method.")

  def untransform(self, z):
    """Reverses stored transformation on provided data."""
    raise NotImplementedError(
        "Each Transformer is responsible for its own untransfomr method.")

  def transform(self, dataset, parallel=False):
    """
    Transforms all internally stored data.
    Adds X-transform, y-transform columns to metadata.
    """
    _, y_shape, w_shape, _ = dataset.get_shape()
    if y_shape == tuple() and self.transform_y:
      raise ValueError("Cannot transform y when y_values are not present")
    if w_shape == tuple() and self.transform_w:
      raise ValueError("Cannot transform w when w_values are not present")
    return dataset.transform(lambda X, y, w: self.transform_array(X, y, w))

  def transform_on_array(self, X, y, w):
    """
    Transforms numpy arrays X, y, and w
    """
    X, y, w = self.transform_array(X, y, w)
    return X, y, w


class NormalizationTransformer(Transformer):

  def __init__(self,
               transform_X=False,
               transform_y=False,
               transform_w=False,
               dataset=None,
               transform_gradients=False):
    """Initialize normalization transformation."""
    if transform_X:
      X_means, X_stds = dataset.get_statistics(X_stats=True, y_stats=False)
      self.X_means = X_means
      self.X_stds = X_stds
    elif transform_y:
      y_means, y_stds = dataset.get_statistics(X_stats=False, y_stats=True)
      self.y_means = y_means
      # Control for pathological case with no variance.
      y_stds = np.array(y_stds)
      y_stds[y_stds == 0] = 1.
      self.y_stds = y_stds
    self.transform_gradients = transform_gradients
    if self.transform_gradients:
      true_grad, ydely_means = get_grad_statistics(dataset)
      self.grad = np.reshape(true_grad, (true_grad.shape[0], -1, 3))
      self.ydely_means = ydely_means

    super(NormalizationTransformer, self).__init__(
        transform_X=transform_X,
        transform_y=transform_y,
        transform_w=transform_w,
        dataset=dataset)

  def transform(self, dataset, parallel=False):
    return super(NormalizationTransformer, self).transform(
        dataset, parallel=parallel)

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    if self.transform_X:
      X = np.nan_to_num((X - self.X_means) / self.X_stds)
    if self.transform_y:
      y = np.nan_to_num((y - self.y_means) / self.y_stds)
    return (X, y, w)

  def untransform(self, z):
    """
    Undo transformation on provided data.
    """
    if self.transform_X:
      return z * self.X_stds + self.X_means
    elif self.transform_y:
      y_stds = self.y_stds
      y_means = self.y_means
      n_tasks = self.y_stds.shape[0]
      z_shape = list(z.shape)
      # Get the reversed shape of z: (..., n_tasks, batch_size)
      z_shape.reverse()
      # Find the task dimension of z
      for dim in z_shape:
        if dim != n_tasks and dim == 1:
          # Prevent broadcasting on wrong dimension
          y_stds = np.expand_dims(y_stds, -1)
          y_means = np.expand_dims(y_means, -1)
      return z * y_stds + y_means

  def untransform_grad(self, grad, tasks):
    """
    Undo transformation on gradient.
    """
    if self.transform_y:

      grad_means = self.y_means[1:]
      energy_var = self.y_stds[0]
      grad_var = 1 / energy_var * (
          self.ydely_means - self.y_means[0] * self.y_means[1:])
      energy = tasks[:, 0]
      transformed_grad = []

      for i in range(energy.size):
        Etf = energy[i]
        grad_Etf = grad[i].flatten()
        grad_E = Etf * grad_var + energy_var * grad_Etf + grad_means
        grad_E = np.reshape(grad_E, (-1, 3))
        transformed_grad.append(grad_E)

      transformed_grad = np.asarray(transformed_grad)
      return transformed_grad


class ClippingTransformer(Transformer):
  """Clip large values in datasets.

     Example:

     >>> n_samples = 10
     >>> n_features = 3
     >>> n_tasks = 1
     >>> ids = np.arange(n_samples)
     >>> X = np.random.rand(n_samples, n_features)
     >>> y = np.zeros((n_samples, n_tasks))
     >>> w = np.ones((n_samples, n_tasks))
     >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
     >>> transformer = dc.trans.ClippingTransformer(transform_X=True)
     >>> dataset = transformer.transform(dataset)

  """

  def __init__(self,
               transform_X=False,
               transform_y=False,
               transform_w=False,
               dataset=None,
               x_max=5.,
               y_max=500.):
    """Initialize clipping transformation.

    Parameters:
    ----------
    transform_X: bool, optional (default False)
      Whether to transform X
    transform_y: bool, optional (default False)
      Whether to transform y
    transform_w: bool, optional (default False)
      Whether to transform w
    dataset: dc.data.Dataset object, optional
      Dataset to be transformed
    x_max: float, optional
      Maximum absolute value for X
    y_max: float, optional
      Maximum absolute value for y

    """
    super(ClippingTransformer, self).__init__(
        transform_X=transform_X,
        transform_y=transform_y,
        transform_w=transform_w,
        dataset=dataset)
    assert not transform_w
    self.x_max = x_max
    self.y_max = y_max

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays.

    Parameters:
    ----------
    X: np.ndarray
      Features
    y: np.ndarray
      Tasks
    w: np.ndarray
      Weights

    Returns:
    -------
    X: np.ndarray
      Transformed features
    y: np.ndarray
      Transformed tasks
    w: np.ndarray
      Transformed weights

    """
    if self.transform_X:
      X[X > self.x_max] = self.x_max
      X[X < (-1.0 * self.x_max)] = -1.0 * self.x_max
    if self.transform_y:
      y[y > self.y_max] = self.y_max
      y[y < (-1.0 * self.y_max)] = -1.0 * self.y_max
    return (X, y, w)

  def untransform(self, z):
    raise NotImplementedError(
        "Cannot untransform datasets with ClippingTransformer.")


class LogTransformer(Transformer):

  def __init__(self,
               transform_X=False,
               transform_y=False,
               features=None,
               tasks=None,
               dataset=None):
    self.features = features
    self.tasks = tasks
    """Initialize log  transformation."""
    super(LogTransformer, self).__init__(
        transform_X=transform_X, transform_y=transform_y, dataset=dataset)

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    if self.transform_X:
      num_features = len(X[0])
      if self.features is None:
        X = np.log(X + 1)
      else:
        for j in range(num_features):
          if j in self.features:
            X[:, j] = np.log(X[:, j] + 1)
          else:
            X[:, j] = X[:, j]
    if self.transform_y:
      num_tasks = len(y[0])
      if self.tasks is None:
        y = np.log(y + 1)
      else:
        for j in range(num_tasks):
          if j in self.tasks:
            y[:, j] = np.log(y[:, j] + 1)
          else:
            y[:, j] = y[:, j]
    return (X, y, w)

  def untransform(self, z):
    """
    Undo transformation on provided data.
    """
    if self.transform_X:
      num_features = len(z[0])
      if self.features is None:
        return np.exp(z) - 1
      else:
        for j in range(num_features):
          if j in self.features:
            z[:, j] = np.exp(z[:, j]) - 1
          else:
            z[:, j] = z[:, j]
        return z
    elif self.transform_y:
      num_tasks = len(z[0])
      if self.tasks is None:
        return np.exp(z) - 1
      else:
        for j in range(num_tasks):
          if j in self.tasks:
            z[:, j] = np.exp(z[:, j]) - 1
          else:
            z[:, j] = z[:, j]
        return z


class BalancingTransformer(Transformer):
  """Balance positive and negative examples for weights."""

  def __init__(self,
               transform_X=False,
               transform_y=False,
               transform_w=False,
               dataset=None,
               seed=None):
    super(BalancingTransformer, self).__init__(
        transform_X=transform_X,
        transform_y=transform_y,
        transform_w=transform_w,
        dataset=dataset)
    # BalancingTransformer can only transform weights.
    assert not transform_X
    assert not transform_y
    assert transform_w

    # Compute weighting factors from dataset.
    y = self.dataset.y
    w = self.dataset.w
    # Ensure dataset is binary
    np.testing.assert_allclose(sorted(np.unique(y)), np.array([0., 1.]))
    weights = []
    for ind, task in enumerate(self.dataset.get_task_names()):
      task_w = w[:, ind]
      task_y = y[:, ind]
      # Remove labels with zero weights
      task_y = task_y[task_w != 0]
      num_positives = np.count_nonzero(task_y)
      num_negatives = len(task_y) - num_positives
      if num_positives > 0:
        pos_weight = float(num_negatives) / num_positives
      else:
        pos_weight = 1
      neg_weight = 1
      weights.append((neg_weight, pos_weight))
    self.weights = weights

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    w_balanced = np.zeros_like(w)
    for ind, task in enumerate(self.dataset.get_task_names()):
      task_y = y[:, ind]
      task_w = w[:, ind]
      zero_indices = np.logical_and(task_y == 0, task_w != 0)
      one_indices = np.logical_and(task_y == 1, task_w != 0)
      w_balanced[zero_indices, ind] = self.weights[ind][0]
      w_balanced[one_indices, ind] = self.weights[ind][1]
    return (X, y, w_balanced)


class CDFTransformer(Transformer):
  """Histograms the data and assigns values based on sorted list."""
  """Acts like a Cumulative Distribution Function (CDF)."""

  def __init__(self, transform_X=False, transform_y=False, dataset=None,
               bins=2):
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.bins = bins
    self.y = dataset.y
    # self.w = dataset.w

  # TODO (flee2): for transform_y, figure out weights

  def transform(self, dataset, bins):
    """Performs CDF transform on data."""
    X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
    w_t = w
    ids_t = ids
    if self.transform_X:
      X_t = get_cdf_values(X, self.bins)
      y_t = y
    if self.transform_y:
      X_t = X
      y_t = get_cdf_values(y, self.bins)
      # print("y will not be transformed by CDFTransformer, for now.")
    return NumpyDataset(X_t, y_t, w_t, ids_t)

  def untransform(self, z):
    # print("Cannot undo CDF Transformer, for now.")
    # Need this for transform_y
    if self.transform_y:
      return self.y


def get_cdf_values(array, bins):
  # array = np.transpose(array)
  n_rows = array.shape[0]
  n_cols = array.shape[1]
  array_t = np.zeros((n_rows, n_cols))
  parts = n_rows / bins
  hist_values = np.zeros(n_rows)
  sorted_hist_values = np.zeros(n_rows)
  for row in range(n_rows):
    if np.remainder(bins, 2) == 1:
      hist_values[row] = np.floor(np.divide(row, parts)) / (bins - 1)
    else:
      hist_values[row] = np.floor(np.divide(row, parts)) / bins
  for col in range(n_cols):
    order = np.argsort(array[:, col], axis=0)
    sorted_hist_values = hist_values[order]
    array_t[:, col] = sorted_hist_values

  return array_t


class PowerTransformer(Transformer):
  """Takes power n transforms of the data based on an input vector."""

  def __init__(self, transform_X=False, transform_y=False, powers=[1]):
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.powers = powers

  def transform(self, dataset):
    """Performs power transform on data."""
    X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
    w_t = w
    ids_t = ids
    n_powers = len(self.powers)
    if self.transform_X:
      X_t = np.power(X, self.powers[0])
      for i in range(1, n_powers):
        X_t = np.hstack((X_t, np.power(X, self.powers[i])))
      y_t = y
    if self.transform_y:
      # print("y will not be transformed by PowerTransformer, for now.")
      y_t = np.power(y, self.powers[0])
      for i in range(1, n_powers):
        y_t = np.hstack((y_t, np.power(y, self.powers[i])))
      X_t = X
    """
    shutil.rmtree(dataset.data_dir)
    os.makedirs(dataset.data_dir)
    DiskDataset.from_numpy(dataset.data_dir, X_t, y_t, w_t, ids_t)
    return dataset
    """
    return NumpyDataset(X_t, y_t, w_t, ids_t)

  def untransform(self, z):
    # print("Cannot undo Power Transformer, for now.")
    n_powers = len(self.powers)
    orig_len = (z.shape[1]) / n_powers
    z = z[:, :orig_len]
    z = np.power(z, 1 / self.powers[0])
    return z


class CoulombFitTransformer(Transformer):
  """Performs randomization and binarization operations on batches of Coulomb Matrix features during fit.

     Example:

     >>> n_samples = 10
     >>> n_features = 3
     >>> n_tasks = 1
     >>> ids = np.arange(n_samples)
     >>> X = np.random.rand(n_samples, n_features, n_features)
     >>> y = np.zeros((n_samples, n_tasks))
     >>> w = np.ones((n_samples, n_tasks))
     >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
     >>> fit_transformers = [dc.trans.CoulombFitTransformer(dataset)]
     >>> model = dc.models.TensorflowMultiTaskFitTransformRegressor(n_tasks,
     ...    [n_features, n_features], batch_size=n_samples, fit_transformers=fit_transformers, n_evals=1)
     n_features after fit_transform: 12
  """

  def __init__(self, dataset):
    """Initializes CoulombFitTransformer.

    Parameters:
    ----------
    dataset: dc.data.Dataset object

    """
    X = dataset.X
    num_atoms = X.shape[1]
    self.step = 1.0
    self.noise = 1.0
    self.triuind = (np.arange(num_atoms)[:, np.newaxis] <=
                    np.arange(num_atoms)[np.newaxis, :]).flatten()
    self.max = 0
    for _ in range(10):
      self.max = np.maximum(self.max, self.realize(X).max(axis=0))
    X = self.expand(self.realize(X))
    self.nbout = X.shape[1]
    self.mean = X.mean(axis=0)
    self.std = (X - self.mean).std()

  def realize(self, X):
    """Randomize features.

    Parameters:
    ----------
    X: np.ndarray
      Features

    Returns:
    -------
    X: np.ndarray
      Randomized features


    """

    def _realize_(x):
      assert (len(x.shape) == 2)
      inds = np.argsort(-(x**2).sum(axis=0)**.5 + np.random.normal(
          0, self.noise, x[0].shape))
      x = x[inds, :][:, inds] * 1
      x = x.flatten()[self.triuind]
      return x

    return np.array([_realize_(z) for z in X])

  def normalize(self, X):
    """Normalize features.

    Parameters:
    ----------
    X: np.ndarray
      Features

    Returns:
    -------
    X: np.ndarray
      Normalized features

    """
    return (X - self.mean) / self.std

  def expand(self, X):
    """Binarize features.

    Parameters:
    ----------
    X: np.ndarray
      Features

    Returns:
    -------
    X: np.ndarray
      Binarized features

    """
    Xexp = []
    for i in range(X.shape[1]):
      for k in np.arange(0, self.max[i] + self.step, self.step):
        Xexp += [np.tanh((X[:, i] - k) / self.step)]
    return np.array(Xexp).T

  def X_transform(self, X):
    """Perform Coulomb Fit transform on features.

    Parameters:
    ----------
    X: np.ndarray
      Features

    Returns:
    -------
    X: np.ndarray
      Transformed features

    """

    X = self.normalize(self.expand(self.realize(X)))
    return X

  def transform_array(self, X, y, w):
    X = self.X_transform(X)
    return (X, y, w)

  def untransform(self, z):
    raise NotImplementedError(
        "Cannot untransform datasets with FitTransformer.")


class IRVTransformer():
  """Performs transform from ECFP to IRV features(K nearest neibours)."""

  def __init__(self, K, n_tasks, dataset, transform_y=False, transform_x=False):
    """Initializes IRVTransformer.
    Parameters:
    ----------
    dataset: dc.data.Dataset object
      train_dataset
    K: int
      number of nearest neighbours being count
    n_tasks: int
      number of tasks

    """
    self.X = dataset.X
    self.n_tasks = n_tasks
    self.K = K
    self.y = dataset.y
    self.w = dataset.w
    self.transform_x = transform_x
    self.transform_y = transform_y

  def realize(self, similarity, y, w):
    """find samples with top ten similarity values in the reference dataset

    Parameters:
    -----------
    similarity: np.ndarray
      similarity value between target dataset and reference dataset
      should have size of (n_samples_in_target, n_samples_in_reference)
    y: np.array
      labels for a single task
    w: np.array
      weights for a single task

    Return:
    ----------
    features: list
      n_samples * np.array of size (2*K,)
      each array includes K similarity values and corresponding labels

    """
    features = []
    similarity_xs = similarity * np.sign(w)
    [target_len, reference_len] = similarity_xs.shape
    g_temp = tf.Graph()
    values = []
    top_labels = []
    with g_temp.as_default():
      with tf.device('/cpu:0'):
        labels_tf = tf.constant(y)
        similarity_placeholder = tf.placeholder(
            dtype=tf.float64, shape=(None, reference_len))
        value, indice = tf.nn.top_k(
            similarity_placeholder, k=self.K + 1, sorted=True)
        # the tf graph here pick up the (K+1) highest similarity values
        # and their indices
        top_label = tf.gather(labels_tf, indice)
      # map the indices to labels
      feed_dict = {}
      with tf.Session() as sess:
        for count in range(target_len // 100 + 1):
          feed_dict[similarity_placeholder] = similarity_xs[count * 100:min((
              count + 1) * 100, target_len), :]
          # generating batch of data by slicing similarity matrix
          # into 100*reference_dataset_length
          fetched_values = sess.run([value, top_label], feed_dict=feed_dict)
          values.append(fetched_values[0])
          top_labels.append(fetched_values[1])
    values = np.concatenate(values, axis=0)
    top_labels = np.concatenate(top_labels, axis=0)
    # concatenate batches of data together
    for count in range(values.shape[0]):
      if values[count, 0] == 1:
        features.append(
            np.concatenate([
                values[count, 1:(self.K + 1)], top_labels[count, 1:(self.K + 1)]
            ]))
        # highest similarity is 1: target is in the reference
        # use the following K points
      else:
        features.append(
            np.concatenate(
                [values[count, 0:self.K], top_labels[count, 0:self.K]]))
        # highest less than 1: target not in the reference, use top K points
    return features

  def X_transform(self, X_target):
    """ Calculate similarity between target dataset(X_target) and
    reference dataset(X): #(1 in intersection)/#(1 in union)
         similarity = (X_target intersect X)/(X_target union X)
    Parameters:
    -----------
    X_target: np.ndarray
      fingerprints of target dataset
      should have same length with X in the second axis

    Returns:
    ----------
    X_target: np.ndarray
      features of size(batch_size, 2*K*n_tasks)

    """
    X_target2 = []
    n_features = X_target.shape[1]
    print('start similarity calculation')
    time1 = time.time()
    similarity = IRVTransformer.matrix_mul(X_target, np.transpose(self.X)) / (
        n_features - IRVTransformer.matrix_mul(1 - X_target,
                                               np.transpose(1 - self.X)))
    time2 = time.time()
    print('similarity calculation takes %i s' % (time2 - time1))
    for i in range(self.n_tasks):
      X_target2.append(self.realize(similarity, self.y[:, i], self.w[:, i]))
    return np.concatenate([z for z in np.array(X_target2)], axis=1)

  @staticmethod
  def matrix_mul(X1, X2, shard_size=5000):
    """ Calculate matrix multiplication for big matrix,
    X1 and X2 are sliced into pieces with shard_size rows(columns)
    then multiplied together and concatenated to the proper size
    """
    X1 = np.float_(X1)
    X2 = np.float_(X2)
    X1_shape = X1.shape
    X2_shape = X2.shape
    assert X1_shape[1] == X2_shape[0]
    X1_iter = X1_shape[0] // shard_size + 1
    X2_iter = X2_shape[1] // shard_size + 1
    all_result = np.zeros((1,))
    for X1_id in range(X1_iter):
      result = np.zeros((1,))
      for X2_id in range(X2_iter):
        partial_result = np.matmul(X1[X1_id * shard_size:min((
            X1_id + 1) * shard_size, X1_shape[0]), :],
                                   X2[:, X2_id * shard_size:min((
                                       X2_id + 1) * shard_size, X2_shape[1])])
        # calculate matrix multiplicatin on slices
        if result.size == 1:
          result = partial_result
        else:
          result = np.concatenate((result, partial_result), axis=1)
        # concatenate the slices together
        del partial_result
      if all_result.size == 1:
        all_result = result
      else:
        all_result = np.concatenate((all_result, result), axis=0)
      del result
    return all_result

  def transform(self, dataset):
    X_length = dataset.X.shape[0]
    X_trans = []
    for count in range(X_length // 5000 + 1):
      X_trans.append(
          self.X_transform(dataset.X[count * 5000:min((count + 1) * 5000,
                                                      X_length), :]))
    X_trans = np.concatenate(X_trans, axis=0)
    return NumpyDataset(X_trans, dataset.y, dataset.w, ids=None)

  def untransform(self, z):
    raise NotImplementedError(
        "Cannot untransform datasets with IRVTransformer.")


class DAGTransformer(Transformer):
  """Performs transform from ConvMol adjacency lists to
  DAG calculation orders
  """

  def __init__(self,
               max_atoms=50,
               transform_X=True,
               transform_y=False,
               transform_w=False):
    """Initializes DAGTransformer.
    Only X can be transformed
    """
    self.max_atoms = max_atoms
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.transform_w = transform_w
    assert self.transform_X
    assert not self.transform_y
    assert not self.transform_w

  def transform_array(self, X, y, w):
    """Add calculation orders to ConvMol objects"""
    if self.transform_X:
      for idm, mol in enumerate(X):
        X[idm].parents = self.UG_to_DAG(mol)
    return (X, y, w)

  def untransform(self, z):
    raise NotImplementedError(
        "Cannot untransform datasets with DAGTransformer.")

  def UG_to_DAG(self, sample):
    """This function generates the DAGs for a molecule
    """
    # list of calculation orders for DAGs
    # stemming from one specific atom in the molecule
    parents = []
    # starting from the adjacency list derived by graphconv featurizer
    UG = sample.get_adjacency_list()
    # number of atoms, also number of DAGs
    n_atoms = sample.get_num_atoms()
    # DAG on a molecule with k atoms includes k steps of calculation,
    # each step calculating graph features for one atom.
    # `max_atoms` is the maximum number of steps
    max_atoms = self.max_atoms
    for count in range(n_atoms):
      # each iteration generates the DAG starting from atom with index `count`
      DAG = []
      # list of lists, elements represent the calculation orders
      # for atoms in the current graph
      parent = [[] for i in range(n_atoms)]
      # starting from the target atom with index `count`
      current_atoms = [count]
      # flags of whether the atom is already included in the DAG
      atoms_indicator = np.zeros((n_atoms,))
      # atom `count` is in the DAG
      radial = 1
      atoms_indicator[count] = radial
      # recording number of radial propagation steps
      while not np.all(atoms_indicator):
        # in the fisrt loop, atoms directly connected to `count` will be added
        # into the DAG(radial=0), then atoms two-bond away from `count`
        # will be added in the second loop(radial=1).
        # atoms i-bond away will be added in i-th loop
        if radial > n_atoms:
          # when molecules have separate parts, starting from one part,
          # it is not possible to include all atoms.
          # this break quit the loop when going into such condition
          break
        # reinitialize targets for next iteration
        next_atoms = []
        radial = radial + 1
        for current_atom in current_atoms:
          for atom_adj in UG[current_atom]:
            # atoms connected to current_atom
            if atoms_indicator[atom_adj] == 0:
              # generate the dependency map of current DAG
              # atoms connected to `current_atoms`(and not included in the DAG)
              # are added, and will be the `current_atoms` for next iteration.
              DAG.append((current_atom, atom_adj))
              atoms_indicator[atom_adj] = radial
              next_atoms.append(atom_adj)
        current_atoms = next_atoms
      # DAG starts from the target atom, calculation should go in reverse
      for edge in reversed(DAG):
        # `edge[1]` is the parent of `edge[0]`
        parent[edge[0]].append(edge[1] % max_atoms)
        parent[edge[0]].extend(parent[edge[1]])

      for i, order in enumerate(parent):
        parent[i] = sorted(order, key=lambda x: atoms_indicator[x])
      # after this loop, `parents[i]` includes all parents of atom i
      for ids, atom in enumerate(parent):
        # manually adding the atom index into its parents list
        parent[ids].insert(0, ids % max_atoms)
      # after this loop, `parents[i][0]` is i, `parents[i][1:]` are all parents of atom i

      # atoms with less parents(farther from the target atom) come first.
      # graph features of atoms without parents will be first calculated,
      # then atoms with more parents can be calculated in order
      # based on previously calculated graph features.
      # target atom of this DAG will be calculated in the last step
      parent = sorted(parent, key=len)

      for ids, atom in enumerate(parent):
        n_par = len(atom)
        # padding with `max_atoms`
        if n_par < max_atoms:
          parent[ids].extend([max_atoms for i in range(max_atoms - n_par)])
        if n_par > max_atoms:
          parent[ids] = parent[ids][:max_atoms]

      if len(parent) > max_atoms:
        parent = parent[-max_atoms:]
      while len(parent) < max_atoms:
        # padding
        parent.insert(0, [max_atoms] * max_atoms)
      # `parents[i]` is the calculation order for the DAG stemming from atom i,
      # which is a max_atoms * max_atoms numpy array after padding
      parents.append(np.array(parent))

    return parents


class ImageTransformer(Transformer):
  """
  Convert an image into width, height, channel
  """

  def __init__(self,
               size,
               transform_X=True,
               transform_y=False,
               transform_w=False):
    """Initializes transformation based on dataset statistics."""
    self.size = size
    self.transform_X = True
    self.transform_y = False
    self.transform_w = False

  def transform_array(self, X, y, w):
    """Transform the data in a set of (X, y, w) arrays."""
    images = [scipy.ndimage.imread(x, mode='RGB') for x in X]
    images = [scipy.misc.imresize(x, size=self.size) for x in images]
    return np.array(images), y, w


class ANITransformer(Transformer):
  """Performs transform from 3D coordinates to ANI symmetry functions
  """

  def __init__(self,
               max_atoms=23,
               radial_cutoff=4.6,
               angular_cutoff=3.1,
               radial_length=32,
               angular_length=8,
               atom_cases=[1, 6, 7, 8, 16],
               atomic_number_differentiated=True,
               coordinates_in_bohr=True,
               transform_X=True,
               transform_y=False,
               transform_w=False):
    """
    Only X can be transformed
    """
    self.max_atoms = max_atoms
    self.radial_cutoff = radial_cutoff
    self.angular_cutoff = angular_cutoff
    self.radial_length = radial_length
    self.angular_length = angular_length
    self.atom_cases = atom_cases
    self.atomic_number_differentiated = atomic_number_differentiated
    self.coordinates_in_bohr = coordinates_in_bohr
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.transform_w = transform_w
    self.compute_graph = self.build()
    self.sess = tf.Session(graph=self.compute_graph)
    self.transform_batch_size = 32
    assert self.transform_X
    assert not self.transform_y
    assert not self.transform_w

  def transform_array(self, X, y, w):
    if self.transform_X:
      n_samples = X.shape[0]

      X_out = []
      num_transformed = 0
      start = 0

      batch_size = self.transform_batch_size

      while True:
        end = min((start + 1) * batch_size, X.shape[0])
        X_batch = X[(start * batch_size):end]
        output = self.sess.run(
            [self.outputs], feed_dict={self.inputs: X_batch})[0]
        X_out.append(output)
        num_transformed = num_transformed + X_batch.shape[0]
        print('%i samples transformed' % num_transformed)
        start += 1
        if end >= len(X):
          break

      X_new = np.concatenate(X_out, axis=0)
      assert X_new.shape[0] == X.shape[0]
    return (X_new, y, w)

  def untransform(self, z):
    raise NotImplementedError(
        "Cannot untransform datasets with ANITransformer.")

  def build(self):
    """ tensorflow computation graph for transform """
    graph = tf.Graph()
    with graph.as_default():
      self.inputs = tf.placeholder(tf.float32, shape=(None, self.max_atoms, 4))
      atom_numbers = tf.cast(self.inputs[:, :, 0], tf.int32)
      flags = tf.sign(atom_numbers)
      flags = tf.to_float(tf.expand_dims(flags, 1) * tf.expand_dims(flags, 2))
      coordinates = self.inputs[:, :, 1:]
      if self.coordinates_in_bohr:
        coordinates = coordinates * 0.52917721092
      d = self.distance_matrix(coordinates, flags)
      d_radial_cutoff = self.distance_cutoff(d, self.radial_cutoff, flags)
      d_angular_cutoff = self.distance_cutoff(d, self.angular_cutoff, flags)
      radial_sym = self.radial_symmetry(d_radial_cutoff, d, atom_numbers)
      angular_sym = self.angular_symmetry(d_angular_cutoff, d, atom_numbers,
                                          coordinates)
      self.outputs = tf.concat(
          [
              tf.to_float(tf.expand_dims(atom_numbers, 2)), radial_sym,
              angular_sym
          ],
          axis=2)
    return graph

  def distance_matrix(self, coordinates, flags):
    """ Generate distance matrix """
    max_atoms = self.max_atoms
    tensor1 = tf.stack([coordinates] * max_atoms, axis=1)
    tensor2 = tf.stack([coordinates] * max_atoms, axis=2)

    # Calculate pairwise distance
    d = tf.sqrt(tf.reduce_sum(tf.square(tensor1 - tensor2), axis=3))
    # Masking for valid atom index
    d = d * flags
    return d

  def distance_cutoff(self, d, cutoff, flags):
    """ Generate distance matrix with trainable cutoff """
    # Cutoff with threshold Rc
    d_flag = flags * tf.sign(cutoff - d)
    d_flag = tf.nn.relu(d_flag)
    d_flag = d_flag * tf.expand_dims((1 - tf.eye(self.max_atoms)), 0)
    d = 0.5 * (tf.cos(np.pi * d / cutoff) + 1)
    return d * d_flag

  def radial_symmetry(self, d_cutoff, d, atom_numbers):
    """ Radial Symmetry Function """
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.radial_cutoff, self.radial_length)
    ita = np.ones_like(Rs) * 3 / (Rs[1] - Rs[0])**2
    Rs = tf.to_float(np.reshape(Rs, (1, 1, 1, -1)))
    ita = tf.to_float(np.reshape(ita, (1, 1, 1, -1)))
    length = ita.get_shape().as_list()[-1]

    d_cutoff = tf.stack([d_cutoff] * length, axis=3)
    d = tf.stack([d] * length, axis=3)

    out = tf.exp(-ita * tf.square(d - Rs)) * d_cutoff
    if self.atomic_number_differentiated:
      out_tensors = []
      for atom_type in self.atom_cases:
        selected_atoms = tf.expand_dims(
            tf.expand_dims(atom_numbers_embedded[:, :, atom_type], axis=1),
            axis=3)
        out_tensors.append(tf.reduce_sum(out * selected_atoms, axis=2))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out, axis=2)

  def angular_symmetry(self, d_cutoff, d, atom_numbers, coordinates):
    """ Angular Symmetry Function """

    max_atoms = self.max_atoms
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.angular_cutoff, self.angular_length)
    ita = 3 / (Rs[1] - Rs[0])**2
    thetas = np.linspace(0., np.pi, self.angular_length)
    zeta = float(self.angular_length**2)

    ita, zeta, Rs, thetas = np.meshgrid(ita, zeta, Rs, thetas)
    zeta = tf.to_float(np.reshape(zeta, (1, 1, 1, 1, -1)))
    ita = tf.to_float(np.reshape(ita, (1, 1, 1, 1, -1)))
    Rs = tf.to_float(np.reshape(Rs, (1, 1, 1, 1, -1)))
    thetas = tf.to_float(np.reshape(thetas, (1, 1, 1, 1, -1)))
    length = zeta.get_shape().as_list()[-1]

    vector_distances = tf.stack([coordinates] * max_atoms, 1) - tf.stack(
        [coordinates] * max_atoms, 2)
    R_ij = tf.stack([d] * max_atoms, axis=3)
    R_ik = tf.stack([d] * max_atoms, axis=2)
    f_R_ij = tf.stack([d_cutoff] * max_atoms, axis=3)
    f_R_ik = tf.stack([d_cutoff] * max_atoms, axis=2)

    # Define angle theta = arccos(R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance))
    vector_mul = tf.reduce_sum(tf.stack([vector_distances]*max_atoms, axis=3) * \
        tf.stack([vector_distances]*max_atoms, axis=2), axis=4)
    vector_mul = vector_mul * tf.sign(f_R_ij) * tf.sign(f_R_ik)
    theta = tf.acos(tf.div(vector_mul, R_ij * R_ik + 1e-5))

    R_ij = tf.stack([R_ij] * length, axis=4)
    R_ik = tf.stack([R_ik] * length, axis=4)
    f_R_ij = tf.stack([f_R_ij] * length, axis=4)
    f_R_ik = tf.stack([f_R_ik] * length, axis=4)
    theta = tf.stack([theta] * length, axis=4)

    out_tensor = tf.pow((1. + tf.cos(theta - thetas))/2., zeta) * \
        tf.exp(-ita * tf.square((R_ij + R_ik)/2. - Rs)) * f_R_ij * f_R_ik * 2

    if self.atomic_number_differentiated:
      out_tensors = []
      for id_j, atom_type_j in enumerate(self.atom_cases):
        for atom_type_k in self.atom_cases[id_j:]:
          selected_atoms = tf.stack([atom_numbers_embedded[:, :, atom_type_j]] * max_atoms, axis=2) * \
                           tf.stack([atom_numbers_embedded[:, :, atom_type_k]] * max_atoms, axis=1)
          selected_atoms = tf.expand_dims(
              tf.expand_dims(selected_atoms, axis=1), axis=4)
          out_tensors.append(
              tf.reduce_sum(out_tensor * selected_atoms, axis=(2, 3)))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out_tensor, axis=(2, 3))

  def get_num_feats(self):
    n_feat = self.outputs.get_shape().as_list()[-1]
    return n_feat
