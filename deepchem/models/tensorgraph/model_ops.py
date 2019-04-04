"""Ops for graph construction.

Large amounts of code borrowed from Keras. Will try to incorporate into
DeepChem properly.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from collections import defaultdict


def _to_tensor(x, dtype):
  x = tf.convert_to_tensor(x)
  if x.dtype != dtype:
    x = tf.cast(x, dtype)
  return x


def ones(shape, dtype=None, name=None):
  """Instantiates an all-ones tensor variable and returns it.

  Parameters
  ----------
  shape: Tuple of integers, shape of returned Tensorflow variable.
  dtype: Tensorflow dtype
  name: String, name of returned Tensorflow variable.

  Returns
  -------
  A Tensorflow variable, filled with `1.0`.
  """
  if dtype is None:
    dtype = tf.float32
  shape = tuple(map(int, shape))
  return tf.Variable(
      tf.constant_initializer(1., dtype=dtype)(shape), dtype, name)


def cast_to_floatx(x):
  """Cast a Numpy array to the default Tensorflow float type.

  Parameters
  ----------
  x: Numpy array.

  Returns
  -------
  The same Numpy array, cast to its new type.
  """
  return np.asarray(x, dtype=np.float32)


def moving_average_update(variable, value, momentum):
  try:
    return moving_averages.assign_moving_average(
        variable, value, momentum, zero_debias=False)
  except TypeError:
    return moving_averages.assign_moving_average(variable, value, momentum)


def int_shape(x):
  """Returns the shape of a Tensorflow tensor or a Tensorflow variable as a tuple of
  integers or None entries.

  Arguments
  ---------
  x: Tensor or variable.

  Returns
  -------
  A tuple of integers (or None entries).
  """
  shape = x.get_shape()
  return tuple([i.__int__() for i in shape])


def concatenate(tensors, axis=-1):
  """Concatenates a list of tensors alongside the specified axis.

  Returns
  -------
  A tensor.
  """
  if axis < 0:
    dims = get_ndim(tensors[0])
    if dims:
      axis = axis % dims
    else:
      axis = 0

  try:
    return tf.concat_v2([x for x in tensors], axis)
  except AttributeError:
    return tf.concat(axis=axis, values=[x for x in tensors])


def _normalize_axis(axis, ndim):
  if isinstance(axis, tuple):
    axis = list(axis)
  if isinstance(axis, list):
    for i, a in enumerate(axis):
      if a is not None and a < 0:
        axis[i] = a % ndim
  else:
    if axis is not None and axis < 0:
      axis = axis % ndim
  return axis


def mean(x, axis=None, keepdims=False):
  """Mean of a tensor, alongside the specified axis.

  Parameters
  ----------
  x: A tensor or variable.
  axis: A list of integer. Axes to compute the mean.
  keepdims: A boolean, whether to keep the dimensions or not.
    If keepdims is False, the rank of the tensor is reduced
    by 1 for each entry in axis. If keepdims is True,
    the reduced dimensions are retained with length 1.

  Returns
  -------
  A tensor with the mean of elements of x.
  """
  axis = _normalize_axis(axis, get_ndim(x))
  if x.dtype.base_dtype == tf.bool:
    x = tf.cast(x, tf.float32)
  return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def dot(x, y):
  """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
  When attempting to multiply a ND tensor
  with a ND tensor, it reproduces the Theano behavior.
  (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))

  Parameters
  ----------
  x: Tensor or variable.
  y: Tensor or variable.

  Returns
  -------
  A tensor, dot product of x and y.
  """
  if get_ndim(x) is not None and (get_ndim(x) > 2 or get_ndim(y) > 2):
    x_shape = []
    for i, s in zip(int_shape(x), tf.unstack(tf.shape(x))):
      if i is not None:
        x_shape.append(i)
      else:
        x_shape.append(s)
    x_shape = tuple(x_shape)
    y_shape = []
    for i, s in zip(int_shape(y), tf.unstack(tf.shape(y))):
      if i is not None:
        y_shape.append(i)
      else:
        y_shape.append(s)
    y_shape = tuple(y_shape)
    y_permute_dim = list(range(get_ndim(y)))
    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
    xt = tf.reshape(x, [-1, x_shape[-1]])
    yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
    return tf.reshape(
        tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
  out = tf.matmul(x, y)
  return out


def get_ndim(x):
  """Returns the number of axes in a tensor, as an integer.

  Parameters
  ----------
  x: Tensor or variable.

  Returns
  -------
  Integer (scalar), number of axes.
  """
  dims = x.get_shape()._dims
  if dims is not None:
    return len(dims)
  return None


def get_dtype(x):
  """Returns the dtype of a Tensorflow tensor or variable, as a string.

  Parameters
  ----------
  x: Tensor or variable.

  Returns
  -------
  String, dtype of `x`.
  """
  return x.dtype.name


def clip(x, min_value, max_value):
  """Element-wise value clipping.

  Returns
  -------
  A tensor.
  """
  if max_value is not None and max_value < min_value:
    max_value = min_value
  min_value = _to_tensor(min_value, x.dtype.base_dtype)
  max_value = _to_tensor(max_value, x.dtype.base_dtype)
  return tf.clip_by_value(x, min_value, max_value)


def epsilon():
  """Returns the value of the fuzz
  factor used in numeric expressions.

  Returns
  -------
  A float.
  """
  return 1e-7


def random_uniform_variable(shape,
                            low,
                            high,
                            dtype=tf.float32,
                            name=None,
                            seed=None):
  """Instantiates an variable filled with
  samples drawn from a uniform distribution and returns it.

  Parameters
  ----------
  shape: Tuple of integers, shape of returned variable.
  low: Float, lower boundary of the output inteval.
  high: Float, upper boundary of the output interval.
  dtype: Tensorflow dtype
  name: String, name of returned variable.
  seed: Integer, random seed.

  Returns
  -------
  A tf.Variable, filled with drawn samples.
  """
  shape = tuple(map(int, shape))
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = tf.random_uniform_initializer(
      low, high, dtype=dtype, seed=seed)(shape)
  return tf.Variable(value, dtype=dtype, name=name)


def random_normal_variable(shape,
                           mean,
                           scale,
                           dtype=tf.float32,
                           name=None,
                           seed=None):
  """Instantiates an Tensorflow variable filled with
  samples drawn from a normal distribution and returns it.

  Parameters
  ----------
  shape: Tuple of integers, shape of returned Tensorflow variable.
  mean: Float, mean of the normal distribution.
  scale: Float, standard deviation of the normal distribution.
  dtype: Tensorflow dtype
  name: String, name of returned Tensorflow variable.
  seed: Integer, random seed.

  Returns
  -------
  A tf.Variable, filled with drawn samples.
  """
  shape = tuple(map(int, shape))
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = tf.random_normal_initializer(
      mean, scale, dtype=dtype, seed=seed)(shape)
  return tf.Variable(value, dtype=dtype, name=name)


def max(x, axis=None, keepdims=False):
  """Maximum value in a tensor.

  Parameters
  ----------
  x: A tensor or variable.
  axis: An integer, the axis to find maximum values.
  keepdims: A boolean, whether to keep the dimensions or not.
      If `keepdims` is `False`, the rank of the tensor is reduced
      by 1. If `keepdims` is `True`,
      the reduced dimension is retained with length 1.

  Returns
  -------
  A tensor with maximum values of `x`.
  """
  axis = _normalize_axis(axis, get_ndim(x))
  return tf.reduce_max(x, axis=axis, keepdims=keepdims)


def l2_normalize(x, axis):
  """Normalizes a tensor wrt the L2 norm alongside the specified axis.

  Parameters
  ----------
  x: input tensor.
  axis: axis along which to perform normalization.

  Returns
  -------
  A tensor.
  """
  if axis < 0:
    axis = axis % len(x.get_shape())
  return tf.nn.l2_normalize(x, dim=axis)


def categorical_crossentropy(output, target, from_logits=False):
  """Categorical crossentropy between an output tensor
  and a target tensor, where the target is a tensor of the same
  shape as the output.

  # TODO(rbharath): Should probably swap this over to tf mode.
  """
  # Note: tf.nn.softmax_cross_entropy_with_logits
  # expects logits, Tensorflow expects probabilities.
  if not from_logits:
    # scale preds so that the class probas of each sample sum to 1
    output /= tf.reduce_sum(
        output, axis=len(output.get_shape()) - 1, keepdims=True)
    # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return -tf.reduce_sum(
        target * tf.log(output), axis=len(output.get_shape()) - 1)
  else:
    try:
      return tf.nn.softmax_cross_entropy_with_logits(
          labels=target, logits=output)
    except TypeError:
      return tf.nn.softmax_cross_entropy_with_logits(
          logits=output, labels=target)


def sparse_categorical_crossentropy(output, target, from_logits=False):
  """Categorical crossentropy between an output tensor
  and a target tensor, where the target is an integer tensor.
  """
  # Note: tf.nn.softmax_cross_entropy_with_logits
  # expects logits, Tensorflow expects probabilities.
  if not from_logits:
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1 - epsilon)
    output = tf.log(output)

  output_shape = output.get_shape()
  targets = cast(flatten(target), 'int64')
  logits = tf.reshape(output, [-1, int(output_shape[-1])])
  try:
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)
  except TypeError:
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)
  if len(output_shape) == 3:
    # if our output includes timesteps we need to reshape
    return tf.reshape(res, tf.shape(output)[:-1])
  else:
    return res


def binary_crossentropy(output, target, from_logits=False):
  """Binary crossentropy between an output tensor and a target tensor.

  # Arguments
      output: A tensor.
      target: A tensor with the same shape as `output`.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.

  # Returns
      A tensor.
  """
  # Note: tf.nn.softmax_cross_entropy_with_logits
  # expects logits, Tensorflow expects probabilities.
  if not from_logits:
    # transform back to logits
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1 - epsilon)
    output = tf.log(output / (1 - output))
  try:
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
  except TypeError:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=target)


def sum(x, axis=None, keepdims=False):
  """Sum of the values in a tensor, alongside the specified axis.

  Parameters
  ----------
  x: A tensor or variable.
  axis: An integer, the axis to sum over.
  keepdims: A boolean, whether to keep the dimensions or not.
    If keepdims is False, the rank of the tensor is reduced
    by 1. If keepdims is True,
    the reduced dimension is retained with length 1.

  Returns
  -------
  A tensor with sum of x.
  """
  axis = _normalize_axis(axis, get_ndim(x))
  return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


# TODO(rbharath): Need to rename this. This makes a variable, not just creates
# a tensor. Confusing with tf.zeros...
def zeros(shape, dtype=tf.float32, name=None):
  """Instantiates an all-zeros variable and returns it.

  Parameters
  ----------
  shape: Tuple of integers, shape of returned Tensorflow variable
  dtype: Tensorflow dtype
  name: String, name of returned Tensorflow variable

  Returns
  -------
  A variable (including Tensorflow metadata), filled with `0.0`.
  """
  shape = tuple(map(int, shape))
  return tf.Variable(
      tf.constant_initializer(0., dtype=dtype)(shape), dtype=dtype, name=name)


def cosine_distances(test, support):
  """Computes pairwise cosine distances between provided tensors

  Parameters
  ----------
  test: tf.Tensor
    Of shape (n_test, n_feat)
  support: tf.Tensor
    Of shape (n_support, n_feat)

  Returns
  -------
  tf.Tensor:
    Of shape (n_test, n_support)
  """
  rnorm_test = tf.rsqrt(tf.reduce_sum(tf.square(test), 1, keepdims=True)) + 1e-7
  rnorm_support = tf.rsqrt(tf.reduce_sum(tf.square(support), 1,
                                         keepdims=True)) + 1e-7
  test_normalized = test * rnorm_test
  support_normalized = support * rnorm_support

  # Transpose for mul
  support_normalized_t = tf.transpose(support_normalized, perm=[1, 0])
  g = tf.matmul(test_normalized, support_normalized_t)  # Gram matrix
  return g


def elu(x, alpha=1.):
  """Exponential linear unit.

  Parameters
  ----------
  x: A tensor or variable to compute the activation function for.
  alpha: A scalar, slope of positive section.

  Returns
  -------
  A tensor.
  """
  res = tf.nn.elu(x)
  if alpha == 1:
    return res
  else:
    return tf.where(x > 0, res, alpha * res)


def relu(x, alpha=0., max_value=None):
  """Rectified linear unit.
  With default values, it returns element-wise `max(x, 0)`.

  Parameters
  ----------
  x: A tensor or variable.
  alpha: A scalar, slope of negative section (default=`0.`).
  max_value: Saturation threshold.

  Returns
  -------
  A tensor.
  """
  if alpha != 0.:
    negative_part = tf.nn.relu(-x)
  x = tf.nn.relu(x)
  if max_value is not None:
    max_value = _to_tensor(max_value, x.dtype.base_dtype)
    zero = _to_tensor(0., x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, max_value)
  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x


def lrelu(alpha=0.01):
  """Create a leaky rectified linear unit function.

  This function returns a new function that implements the LReLU with a
  specified alpha.  The returned value can be used as an activation function in
  network layers.

  Parameters
  ----------
  alpha: float
    the slope of the function when x<0

  Returns
  -------
  a function f(x) that returns alpha*x when x<0, and x when x>0.
  """

  def eval(x):
    return relu(x, alpha=alpha)

  return eval


def selu(x):
  """Scaled Exponential Linear unit.

  Parameters
  ----------
  x: A tensor or variable.

  Returns
  -------
  A tensor.

  References
  ----------
  - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * elu(x, alpha)


def hard_sigmoid(x):
  """Segment-wise linear approximation of sigmoid.

  Faster than sigmoid. Returns 0. if x < -2.5, 1. if x > 2.5.
  In -2.5 <= x <= 2.5, returns 0.2 * x + 0.5.

  Parameters
  ----------
  x: A tensor or variable.

  Returns
  -------
  A tensor.
  """
  x = (0.2 * x) + 0.5
  zero = _to_tensor(0., x.dtype.base_dtype)
  one = _to_tensor(1., x.dtype.base_dtype)
  x = tf.clip_by_value(x, zero, one)
  return x


def sqrt(x):
  """Element-wise square root.

  Parameters
  ----------
  x: input tensor.

  Returns
  -------
  A tensor.
  """
  zero = _to_tensor(0., x.dtype.base_dtype)
  inf = _to_tensor(np.inf, x.dtype.base_dtype)
  x = tf.clip_by_value(x, zero, inf)
  return tf.sqrt(x)


def var(x, axis=None, keepdims=False):
  """Variance of a tensor, alongside the specified axis.

  Parameters
  ----------
  x: A tensor or variable.
  axis: An integer, the axis to compute the variance.
  keepdims: A boolean, whether to keep the dimensions or not.
      If keepdims is False, the rank of the tensor is reduced
      by 1. If keepdims is True,
      the reduced dimension is retained with length 1.

  Returns
  -------
  A tensor with the variance of elements of `x`.
  """
  axis = _normalize_axis(axis, get_ndim(x))
  if x.dtype.base_dtype == tf.bool:
    x = tf.cast(x, tf.float32)
  m = tf.reduce_mean(x, axis=axis, keepdims=True)
  devs_squared = tf.square(x - m)
  return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def euclidean_distance(test, support, max_dist_sq=20):
  """Computes pairwise euclidean distances between provided tensors

  TODO(rbharath): BROKEN! THIS DOESN'T WORK!

  Parameters
  ----------
  test: tf.Tensor
    Of shape (n_test, n_feat)
  support: tf.Tensor
    Of shape (n_support, n_feat)
  max_dist_sq: float, optional
    Maximum pairwise distance allowed.

  Returns
  -------
  tf.Tensor:
    Of shape (n_test, n_support)
  """
  test = tf.expand_dims(test, 1)
  support = tf.expand_dims(support, 0)
  g = -tf.maximum(tf.reduce_sum(tf.square(test - support), 2), max_dist_sq)
  return g


def add_bias(tensor, init=None, name=None):
  """Add a bias term to a tensor.

  Parameters
  ----------
  tensor: tf.Tensor
    Variable tensor.
  init: float
    Bias initializer. Defaults to zero.
  name: str
    Name for this op. Defaults to tensor.op.name.

  Returns
  -------
  tf.Tensor
    A biased tensor with the same shape as the input tensor.
  """
  if init is None:
    init = tf.zeros([tensor.get_shape()[-1].value])
  with tf.name_scope(name, tensor.op.name, [tensor]):
    b = tf.Variable(init, name='b')
    return tf.nn.bias_add(tensor, b)


def dropout(tensor, dropout_prob, training=True, training_only=True):
  """Random dropout.

  This implementation supports "always-on" dropout (training_only=False), which
  can be used to calculate model uncertainty. See Gal and Ghahramani,
  http://arxiv.org/abs/1506.02142.

  NOTE(user): To simplify the implementation, I have chosen not to reverse
    the scaling that occurs in tf.nn.dropout when using dropout during
    inference. This shouldn't be an issue since the activations will be scaled
    by the same constant in both training and inference. This means that there
    are no training-time differences between networks that use dropout during
    inference and those that do not.

  Parameters
  ----------
  tensor: tf.Tensor
    Input tensor.
  dropout_prob: float
    Float giving dropout probability for weights (NOT keep probability).
  training_only: bool
    Boolean. If True (standard dropout), apply dropout only
    during training. If False, apply dropout during inference as well.

  Returns
  -------
  tf.Tensor:
    A tensor with the same shape as the input tensor.
  """
  if not dropout_prob:
    return tensor  # do nothing
  keep_prob = 1.0 - dropout_prob
  if training or not training_only:
    tensor = tf.nn.dropout(tensor, keep_prob)
  return tensor


def fully_connected_layer(tensor,
                          size=None,
                          weight_init=None,
                          bias_init=None,
                          name=None):
  """Fully connected layer.

  Parameters
  ----------
  tensor: tf.Tensor
    Input tensor.
  size: int
    Number of output nodes for this layer.
  weight_init: float
    Weight initializer.
  bias_init: float
    Bias initializer.
  name: str
    Name for this op. Defaults to 'fully_connected'.

  Returns
  -------
  tf.Tensor:
    A new tensor representing the output of the fully connected layer.

  Raises
  ------
  ValueError
    If input tensor is not 2D.
  """
  if weight_init is None:
    num_features = tensor.get_shape()[-1].value
    weight_init = tf.truncated_normal([num_features, size], stddev=0.01)
  if bias_init is None:
    bias_init = tf.zeros([size])

  with tf.name_scope(name, 'fully_connected', [tensor]):
    w = tf.Variable(weight_init, name='w', dtype=tf.float32)
    b = tf.Variable(bias_init, name='b', dtype=tf.float32)
    return tf.nn.xw_plus_b(tensor, w, b)


def weight_decay(penalty_type, penalty):
  """Add weight decay.

  Args:
    model: TensorflowGraph.

  Returns:
    A scalar tensor containing the weight decay cost.

  Raises:
    NotImplementedError: If an unsupported penalty type is requested.
  """
  variables = []
  # exclude bias variables
  for v in tf.trainable_variables():
    if v.get_shape().ndims == 2:
      variables.append(v)

  with tf.name_scope('weight_decay'):
    if penalty_type == 'l1':
      cost = tf.add_n([tf.reduce_sum(tf.abs(v)) for v in variables])
    elif penalty_type == 'l2':
      cost = tf.add_n([tf.nn.l2_loss(v) for v in variables])
    else:
      raise NotImplementedError('Unsupported penalty_type %s' % penalty_type)
    cost *= penalty
    #tf.scalar_summary('Weight Decay Cost', cost)
  return cost


def multitask_logits(features,
                     num_tasks,
                     num_classes=2,
                     weight_init=None,
                     bias_init=None,
                     dropout_prob=None,
                     name=None):
  """Create a logit tensor for each classification task.

  Args:
    features: A 2D tensor with dimensions batch_size x num_features.
    num_tasks: Number of classification tasks.
    num_classes: Number of classes for each task.
    weight_init: Weight initializer.
    bias_init: Bias initializer.
    dropout_prob: Float giving dropout probability for weights (NOT keep
      probability).
    name: Name for this op. Defaults to 'multitask_logits'.

  Returns:
    A list of logit tensors; one for each classification task.
  """
  logits_list = []
  with tf.name_scope('multitask_logits'):
    for task_idx in range(num_tasks):
      with tf.name_scope(name,
                         ('task' + str(task_idx).zfill(len(str(num_tasks)))),
                         [features]):
        logits_list.append(
            logits(
                features,
                num_classes,
                weight_init=weight_init,
                bias_init=bias_init,
                dropout_prob=dropout_prob))
  return logits_list


def logits(features,
           num_classes=2,
           weight_init=None,
           bias_init=None,
           dropout_prob=None,
           name=None):
  """Create a logits tensor for a single classification task.

  You almost certainly don't want dropout on there -- it's like randomly setting
  the (unscaled) probability of a target class to 0.5.

  Args:
    features: A 2D tensor with dimensions batch_size x num_features.
    num_classes: Number of classes for each task.
    weight_init: Weight initializer.
    bias_init: Bias initializer.
    dropout_prob: Float giving dropout probability for weights (NOT keep
      probability).
    name: Name for this op.

  Returns:
    A logits tensor with shape batch_size x num_classes.
  """
  with tf.name_scope(name, 'logits', [features]) as name:
    return dropout(
        fully_connected_layer(
            features,
            num_classes,
            weight_init=weight_init,
            bias_init=bias_init,
            name=name), dropout_prob)


def softmax_N(tensor, name=None):
  """Apply softmax across last dimension of a tensor.

  Args:
    tensor: Input tensor.
    name: Name for this op. If None, defaults to 'softmax_N'.

  Returns:
    A tensor with softmax-normalized values on the last dimension.
  """
  with tf.name_scope(name, 'softmax_N', [tensor]):
    exp_tensor = tf.exp(tensor)
    reduction_indices = [tensor.get_shape().ndims - 1]
    return tf.math.divide(
        exp_tensor,
        tf.reduce_sum(exp_tensor, axis=reduction_indices, keepdims=True))


def optimizer(optimizer="adam", learning_rate=.001, momentum=.9):
  """Create model optimizer.

  Parameters
  ----------
  optimizer: str, optional
    Name of optimizer
  learning_rate: float, optional
    Learning rate for algorithm
  momentum: float, optional
    Momentum rate

  Returns
  -------
    A training Optimizer.

  Raises:
    NotImplementedError: If an unsupported optimizer is requested.
  """
  # TODO(user): gradient clipping (see Minimize)
  if optimizer == 'adagrad':
    train_op = tf.train.AdagradOptimizer(learning_rate)
  elif optimizer == 'adam':
    train_op = tf.train.AdamOptimizer(learning_rate)
  elif optimizer == 'momentum':
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum)
  elif optimizer == 'rmsprop':
    train_op = tf.train.RMSPropOptimizer(learning_rate, momentum)
  elif optimizer == 'sgd':
    train_op = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise NotImplementedError('Unsupported optimizer %s' % optimizer)
  return train_op
