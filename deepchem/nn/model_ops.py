"""Ops for graph construction.

Large amounts of code borrowed from Keras.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import traceback
import tensorflow as tf
from tensorflow.python.training import moving_averages
from collections import defaultdict

# TODO(rbharath): REMOVE GLOBAL VARS! BREAKS DEEPCHEM STYLE! 
# This is the default internal TF session used by Keras.
# It can be set manually via `set_session(sess)`.
_SESSION = None
_FLOATX = 'float32'
_EPSILON = 10e-8
# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False
_UID_PREFIXES = defaultdict(int)
# This dictionary holds a mapping {graph: learning_phase}.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = {}

# TODO(rbharath): Can this be improved.
def _convert_string_dtype(dtype):
  if dtype == 'float16':
    return tf.float16
  if dtype == 'float32':
    return tf.float32
  elif dtype == 'float64':
    return tf.float64
  elif dtype == 'int16':
    return tf.int16
  elif dtype == 'int32':
    return tf.int32
  elif dtype == 'int64':
    return tf.int64
  elif dtype == 'uint8':
    return tf.int8
  elif dtype == 'uint16':
    return tf.uint16
  else:
    raise ValueError('Unsupported dtype:', dtype)

def learning_phase():
  """Returns the learning phase flag.

  The learning phase flag is a bool tensor (0 = test, 1 = train)
  to be passed as input to any Keras function
  that uses a different behavior at train time and test time.
  """
  graph = tf.get_default_graph()
  if graph not in _GRAPH_LEARNING_PHASES:
    phase = tf.placeholder(dtype='bool',
                           name='keras_learning_phase')
    _GRAPH_LEARNING_PHASES[graph] = phase
  return _GRAPH_LEARNING_PHASES[graph]

def in_train_phase(x, alt):
  """Selects `x` in train phase, and `alt` otherwise.
  Note that `alt` should have the *same shape* as `x`.

  # Returns
      Either `x` or `alt` based on `K.learning_phase`.
  """
  if learning_phase() is 1:
    return x
  elif learning_phase() is 0:
    return alt
  # else: assume learning phase is a placeholder tensor.
  x = switch(learning_phase(), x, alt)
  x._uses_learning_phase = True
  return x

def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
  """Computes mean and std for batch then apply batch_normalization on batch.

  # Returns
      A tuple length of 3, `(normalized_tensor, mean, variance)`.
  """
  mean, var = tf.nn.moments(x, reduction_axes,
                            shift=None, name=None, keep_dims=False)
  if sorted(reduction_axes) == range(ndim(x))[:-1]:
    normed = tf.nn.batch_normalization(x, mean, var,
                                       beta, gamma,
                                       epsilon)
  else:
    # need broadcasting
    target_shape = []
    for axis in range(ndim(x)):
        if axis in reduction_axes:
            target_shape.append(1)
        else:
            target_shape.append(tf.shape(x)[axis])
    target_shape = stack(target_shape)

    broadcast_mean = tf.reshape(mean, target_shape)
    broadcast_var = tf.reshape(var, target_shape)
    broadcast_gamma = tf.reshape(gamma, target_shape)
    broadcast_beta = tf.reshape(beta, target_shape)
    normed = tf.nn.batch_normalization(x, broadcast_mean, broadcast_var,
                                       broadcast_beta, broadcast_gamma,
                                       epsilon)
  return normed, mean, var

def eval(x):
  """Evaluates the value of a variable.
  Returns a Numpy array.

  # Arguments
      x: A variable.

  # Returns
      A Numpy array.

  # Examples
  ```python
      >>> from keras import backend as K
      >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
      >>> K.eval(kvar)
      array([[ 1.,  2.],
             [ 3.,  4.]], dtype=float32)
  ```
  """
  return to_dense(x).eval(session=get_session())

def ones(shape, dtype=None, name=None):
  """Instantiates an all-ones tensor variable and returns it.

  # Arguments
      shape: Tuple of integers, shape of returned Keras variable.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  # Returns
      A Keras variable, filled with `1.0`.

  # Example
  ```python
      >>> from keras import backend as K
      >>> kvar = K.ones((3,4))
      >>> K.eval(kvar)
      array([[ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  1.]], dtype=float32)
  ```
  """
  if dtype is None:
    dtype = tf.float32 
  shape = tuple(map(int, shape))
  tf_dtype = _convert_string_dtype(dtype)
  return variable(tf.constant_initializer(1., dtype=tf_dtype)(shape),
                  dtype, name)

def cast_to_floatx(x):
  """Cast a Numpy array to the default Keras float type.

  # Arguments
      x: Numpy array.

  # Returns
      The same Numpy array, cast to its new type.

  # Example
  ```python
      >>> from keras import backend as K
      >>> K.floatx()
      'float32'
      >>> arr = numpy.array([1.0, 2.0], dtype='float64')
      >>> arr.dtype
      dtype('float64')
      >>> new_arr = K.cast_to_floatx(arr)
      >>> new_arr
      array([ 1.,  2.], dtype=float32)
      >>> new_arr.dtype
      dtype('float32')
  ```
  """
  return np.asarray(x, dtype=_FLOATX)

def to_dense(tensor):
  """Converts a sparse tensor into a dense tensor
  and returns it.

  # Arguments
      tensor: A tensor instance (potentially sparse).

  # Returns
      A dense tensor.

  # Examples
  ```python
      >>> from keras import backend as K
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
      >>> c = K.to_dense(b)
      >>> print(K.is_sparse(c))
      False
  ```
  """
  if is_sparse(tensor):
    return tf.sparse_tensor_to_dense(tensor)
  else:
    return tensor

def moving_average_update(variable, value, momentum):
  try:
    return moving_averages.assign_moving_average(
        variable, value, momentum, zero_debias=False)
  except TypeError:
    return moving_averages.assign_moving_average(
        variable, value, momentum)

def is_sparse(tensor):
  """Returns whether a tensor is a sparse tensor.

  # Arguments
      tensor: A tensor instance.

  # Returns
      A boolean.

  # Example
  ```python
      >>> from keras import backend as K
      >>> a = K.placeholder((2, 2), sparse=False)
      >>> print(K.is_sparse(a))
      False
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
  ```
  """
  return isinstance(tensor, tf.SparseTensor)

def int_shape(x):
  """Returns the shape of a Keras tensor or a Keras variable as a tuple of
  integers or None entries.

  # Arguments
    x: Tensor or variable.

  # Returns
    A tuple of integers (or None entries).

  # Examples
  ```python
      >>> from keras import backend as K
      >>> input = K.placeholder(shape=(2, 4, 5))
      >>> K.int_shape(input)
      (2, 4, 5)
      >>> val = np.array([[1, 2], [3, 4]])
      >>> kvar = K.variable(value=val)
      >>> K.int_shape(kvar)
      (2, 2)
  ```
  """
  shape = x.get_shape()
  return tuple([i.__int__() for i in shape])

def get_value(x):
  """Returns the value of a variable.

  # Arguments
      x: input variable.

  # Returns
      A Numpy array.
  """
  return x.eval(session=get_session())

def get_uid(prefix=''):
  """Provides a unique UID given a string prefix.

  # Arguments
      prefix: string.

  # Returns
      An integer.

  # Example
  ```
      >>> keras.backend.get_uid('dense')
      >>> 1
      >>> keras.backend.get_uid('dense')
      >>> 2
  ```

  """
  _UID_PREFIXES[prefix] += 1
  return _UID_PREFIXES[prefix]

def batch_get_value(xs):
  """Returns the value of more than one tensor variable.

  # Arguments
      x: list of variables.

  # Returns
      A list of Numpy arrays.
  """
  if xs:
    return get_session().run(xs)
  else:
    return []

def batch_set_value(tuples):
  """Sets the values of many tensor variables at once.
  It returns `None`.

  # Arguments
      tuples: a list of tuples `(tensor, value)`.
          `value` should be a Numpy array.
  """
  if tuples:
    assign_ops = []
    feed_dict = {}
    for x, value in tuples:
      value = np.asarray(value)
      tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
      if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
      else:
        assign_placeholder = tf.placeholder(tf_dtype,
                                            shape=value.shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
      assign_ops.append(assign_op)
      feed_dict[assign_placeholder] = value
    get_session().run(assign_ops, feed_dict=feed_dict)

# TODO(rbharath): DANGEROUS! THIS IS LEAKY AND BREAKS DEEPCHEM STYLE!
def get_session():
  """Returns the TF session to be used by the backend.

  If a default TensorFlow session is available, we will return it.

  Else, we will return the global Keras session.

  If no global Keras session exists at this point:
  we will create a new global session.

  Note that you can manually set the global session
  via `K.set_session(sess)`.

  # Returns
      A TensorFlow session.
  """
  global _SESSION
  if tf.get_default_session() is not None:
    session = tf.get_default_session()
  else:
    if _SESSION is None:
      if not os.environ.get('OMP_NUM_THREADS'):
        config = tf.ConfigProto(allow_soft_placement=True)
      else:
        nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
        config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                allow_soft_placement=True)
      _SESSION = tf.Session(config=config)
    session = _SESSION
  if not _MANUAL_VAR_INIT:
    _initialize_variables()
  return session

def concatenate(tensors, axis=-1):
  """Concatenates a list of tensors alongside the specified axis.

  # Returns
      A tensor.
  """
  if axis < 0:
    dims = get_ndim(tensors[0])
    if dims:
      axis = axis % dims
    else:
      axis = 0

  if py_all([is_sparse(x) for x in tensors]):
    return tf.sparse_concat(axis, tensors)
  else:
    try:
      return tf.concat_v2([to_dense(x) for x in tensors], axis)
    except AttributeError:
      return tf.concat(axis, [to_dense(x) for x in tensors])

def mean(x, axis=None, keepdims=False):
  """Mean of a tensor, alongside the specified axis.

  # Arguments
      x: A tensor or variable.
      axis: A list of integer. Axes to compute the mean.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1 for each entry in `axis`. If `keep_dims` is `True`,
          the reduced dimensions are retained with length 1.

  # Returns
      A tensor with the mean of elements of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  if x.dtype.base_dtype == tf.bool:
    x = tf.cast(x, floatx())
  return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def dot(x, y):
  """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
  When attempting to multiply a ND tensor
  with a ND tensor, it reproduces the Theano behavior.
  (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))

  # Arguments
      x: Tensor or variable.
      y: Tensor or variable.

  # Returns
      A tensor, dot product of `x` and `y`.
  """
  if get_ndim(x) is not None and (get_ndim(x) > 2 or get_ndim(y) > 2):
    x_shape = []
    for i, s in zip(int_shape(x), tf.unpack(tf.shape(x))):
      if i is not None:
        x_shape.append(i)
      else:
        x_shape.append(s)
    x_shape = tuple(x_shape)
    y_shape = []
    for i, s in zip(int_shape(y), tf.unpack(tf.shape(y))):
      if i is not None:
        y_shape.append(i)
      else:
        y_shape.append(s)
    y_shape = tuple(y_shape)
    y_permute_dim = list(range(get_ndim(y)))
    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
    xt = tf.reshape(x, [-1, x_shape[-1]])
    yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
    return tf.reshape(tf.matmul(xt, yt),
                      x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
  if is_sparse(x):
    out = tf.sparse_tensor_dense_matmul(x, y)
  else:
    out = tf.matmul(x, y)
  return out

def get_ndim(x):
  """Returns the number of axes in a tensor, as an integer.

  # Arguments
      x: Tensor or variable.

  # Returns
      Integer (scalar), number of axes.
  """
  dims = x.get_shape()._dims
  if dims is not None:
    return len(dims)
  return None

def get_dtype(x):
  """Returns the dtype of a Keras tensor or variable, as a string.

  # Arguments
      x: Tensor or variable.

  # Returns
      String, dtype of `x`.
  """
  return x.dtype.name

def clip(x, min_value, max_value):
  """Element-wise value clipping.

  # Returns
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

  # Returns
      A float.
  """
  return _EPSILON

def variable(value, dtype=tf.float32, name=None):
  """Instantiates a variable and returns it.

  # Arguments
      value: Numpy array, initial value of the tensor.
      dtype: Tensor type.
      name: Optional name string for the tensor.

  # Returns
      A variable instance (with Keras metadata included).
  """
  v = tf.Variable(value, dtype=_convert_string_dtype(dtype), name=name)
  if hasattr(value, 'get_shape'):
    v._keras_shape = tuple(map(int, value.get_shape()))
  v._uses_learning_phase = False
  return v

def random_uniform_variable(shape, low, high, dtype=tf.float32,
                            name=None, seed=None):
  """Instantiates an Keras variable filled with
  samples drawn from a uniform distribution and returns it.

  # Arguments
    shape: Tuple of integers, shape of returned Keras variable.
    low: Float, lower boundary of the output inteval.
    high: Float, upper boundary of the output interval.
    dtype: String, dtype of returned Keras variable.
    name: String, name of returned Keras variable.
    seed: Integer, random seed.

  # Returns
      A Keras variable, filled with drawn samples.
  """
  shape = tuple(map(int, shape))
  tf_dtype = _convert_string_dtype(dtype)
  if seed is None:
      # ensure that randomness is conditioned by the Numpy RNG
      seed = np.random.randint(10e8)
  value = tf.random_uniform_initializer(
      low, high, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)

def random_normal_variable(shape, mean, scale, dtype=tf.float32,
                           name=None, seed=None):
  """Instantiates an Keras variable filled with
  samples drawn from a normal distribution and returns it.

  # Arguments
      shape: Tuple of integers, shape of returned Keras variable.
      mean: Float, mean of the normal distribution.
      scale: Float, standard deviation of the normal distribution.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed.

  # Returns
      A Keras variable, filled with drawn samples.
  """
  shape = tuple(map(int, shape))
  tf_dtype = _convert_string_dtype(dtype)
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = tf.random_normal_initializer(
      mean, scale, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)

def max(x, axis=None, keepdims=False):
  """Maximum value in a tensor.

  # Arguments
    x: A tensor or variable.
    axis: An integer, the axis to find maximum values.
    keepdims: A boolean, whether to keep the dimensions or not.
        If `keepdims` is `False`, the rank of the tensor is reduced
        by 1. If `keepdims` is `True`,
        the reduced dimension is retained with length 1.

  # Returns
    A tensor with maximum values of `x`.
  """
  axis = _normalize_axis(axis, get_ndim(x))
  return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)

def sum(x, axis=None, keepdims=False):
  """Sum of the values in a tensor, alongside the specified axis.

  # Arguments
    x: A tensor or variable.
    axis: An integer, the axis to sum over.
    keepdims: A boolean, whether to keep the dimensions or not.
      If `keepdims` is `False`, the rank of the tensor is reduced
      by 1. If `keepdims` is `True`,
      the reduced dimension is retained with length 1.

  # Returns
    A tensor with sum of `x`.
  """
  axis = _normalize_axis(axis, get_ndim(x))
  return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)

# TODO(rbharath): Need to rename this. This makes a variable, not just creates
# a tensor. Confusing with tf.zeros...
def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.zeros((3,4))
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    shape = tuple(map(int, shape))
    tf_dtype = _convert_string_dtype(dtype)
    return variable(tf.constant_initializer(0., dtype=tf_dtype)(shape),
                    dtype, name)

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
  rnorm_test = tf.rsqrt(tf.reduce_sum(tf.square(test), 1,
                     keep_dims=True)) + 1e-7 
  rnorm_support = tf.rsqrt(tf.reduce_sum(tf.square(support), 1,
                           keep_dims=True)) + 1e-7 
  test_normalized = test * rnorm_test
  support_normalized = support * rnorm_support

  # Transpose for mul
  support_normalized_t = tf.transpose(support_normalized, perm=[1,0])  
  g = tf.matmul(test_normalized, support_normalized_t)  # Gram matrix
  return g

def elu(x, alpha=1.):
  """Exponential linear unit.

  # Arguments
      x: A tenor or variable to compute the activation function for.
      alpha: A scalar, slope of positive section.

  # Returns
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

  # Arguments
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: Saturation threshold.

  # Returns
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

def hard_sigmoid(x):
  """Segment-wise linear approximation of sigmoid.
  Faster than sigmoid.
  Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
  In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

  # Arguments
      x: A tensor or variable.

  # Returns
      A tensor.
  """
  x = (0.2 * x) + 0.5
  zero = _to_tensor(0., x.dtype.base_dtype)
  one = _to_tensor(1., x.dtype.base_dtype)
  x = tf.clip_by_value(x, zero, one)
  return x

def sqrt(x):
  """Element-wise square root.

  # Arguments
      x: input tensor.

  # Returns
      A tensor.
  """
  zero = _to_tensor(0., x.dtype.base_dtype)
  inf = _to_tensor(np.inf, x.dtype.base_dtype)
  x = tf.clip_by_value(x, zero, inf)
  return tf.sqrt(x)

def var(x, axis=None, keepdims=False):
  """Variance of a tensor, alongside the specified axis.

  # Arguments
      x: A tensor or variable.
      axis: An integer, the axis to compute the variance.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  # Returns
      A tensor with the variance of elements of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  if x.dtype.base_dtype == tf.bool:
    x = tf.cast(x, floatx())
  m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
  devs_squared = tf.square(x - m)
  return tf.reduce_mean(devs_squared,
                        reduction_indices=axis,
                        keep_dims=keepdims)

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
  with tf.op_scope([tensor], name, tensor.op.name):
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


def fully_connected_layer(tensor, size=None, weight_init=None, bias_init=None,
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
  if len(tensor.get_shape()) != 2:
    raise ValueError('Dense layer input must be 2D, not %dD'
                     % len(tensor.get_shape()))
  if weight_init is None:
    num_features = tensor.get_shape()[-1].value
    weight_init = tf.truncated_normal([num_features, size], stddev=0.01)
  if bias_init is None:
    bias_init = tf.zeros([size])

  with tf.op_scope([tensor], name, 'fully_connected'):
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
    tf.scalar_summary('Weight Decay Cost', cost)
  return cost


def multitask_logits(features, num_tasks, num_classes=2, weight_init=None,
                     bias_init=None, dropout_prob=None, name=None):
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
      with tf.op_scope([features], name,
                       ('task' + str(task_idx).zfill(len(str(num_tasks))))):
        logits_list.append(
            logits(features, num_classes, weight_init=weight_init,
                   bias_init=bias_init, dropout_prob=dropout_prob))
  return logits_list


def logits(features, num_classes=2, weight_init=None, bias_init=None,
           dropout_prob=None, name=None):
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
  with tf.op_scope([features], name, 'logits') as name:
    return dropout(
        fully_connected_layer(features, num_classes, weight_init=weight_init,
                              bias_init=bias_init, name=name),
        dropout_prob)


def softmax_N(tensor, name=None):
  """Apply softmax across last dimension of a tensor.

  Args:
    tensor: Input tensor.
    name: Name for this op. If None, defaults to 'softmax_N'.

  Returns:
    A tensor with softmax-normalized values on the last dimension.
  """
  with tf.op_scope([tensor], name, 'softmax_N'):
    exp_tensor = tf.exp(tensor)
    reduction_indices = [tensor.get_shape().ndims - 1]
    return tf.div(exp_tensor,
                  tf.reduce_sum(exp_tensor,
                                reduction_indices=reduction_indices,
                                keep_dims=True))

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
    train_op = tf.train.MomentumOptimizer(learning_rate,
                                          momentum)
  elif optimizer == 'rmsprop':
    train_op = tf.train.RMSPropOptimizer(learning_rate,
                                         momentum)
  elif optimizer == 'sgd':
    train_op = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise NotImplementedError('Unsupported optimizer %s' % optimizer)
  return train_op
