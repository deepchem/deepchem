"""
Copies Classes from keras to remove dependency.

Most of this code is copied over from Keras. Hoping to use as a staging
area while we remove our Keras dependency.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
from deepchem.nn import initializations
from deepchem.nn import regularizers
from deepchem.nn import activations
from deepchem.nn import model_ops


def to_list(x):
  """This normalizes a list/tensor into a list.

  If a tensor is passed, we return
  a list of size 1 containing the tensor.
  """
  if isinstance(x, list):
    return x
  return [x]


def object_list_uid(object_list):
  object_list = to_list(object_list)
  return ', '.join([str(abs(id(x))) for x in object_list])


class Layer(object):
  """Abstract base layer class.

  Attributes
  ----------
  name: String, must be unique within a model.
  trainable: Boolean, whether the layer weights
      will be updated during training.
  uses_learning_phase: Whether any operation
      of the layer uses model_ops.in_training_phase()
      or model_ops.in_test_phase().
  input_shape: Shape tuple. Provided for convenience,
    but note that there may be cases in which this
    attribute is ill-defined (e.g. a shared layer
    with multiple input shapes), in which case
    requesting input_shape will raise an Exception.
    Prefer using layer.get_input_shape_for(input_shape),
  output_shape: Shape tuple. See above.
  input, output: Input/output tensor(s). Note that if the layer is used
    more than once (shared layer), this is ill-defined
    and will raise an exception. In such cases, use

  Methods
  -------
  call(x): Where the layer's logic lives.
  __call__(x): Wrapper around the layer logic (`call`).
      If x is a tensor:
          - Connect current layer with last layer from tensor:
          - Add layer to tensor history
      If layer is not built:
  """

  def __init__(self, **kwargs):
    # These properties should have been set
    # by the child class, as appropriate.
    if not hasattr(self, 'uses_learning_phase'):
      self.uses_learning_phase = False

    if not hasattr(self, 'losses'):
      self.losses = []

    # These properties should be set by the user via keyword arguments.
    # note that 'input_dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {
        'input_shape', 'batch_input_shape', 'input_dtype', 'name', 'trainable'
    }
    for kwarg in kwargs.keys():
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)
    name = kwargs.get('name')
    if not name:
      prefix = self.__class__.__name__.lower()
      name = prefix + '_' + str(model_ops.get_uid(prefix))
    self.name = name

    self.trainable = kwargs.get('trainable', True)
    if 'batch_input_shape' in kwargs or 'input_shape' in kwargs:
      # In this case we will create an input layer
      # to insert before the current layer
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        batch_input_shape = (None,) + tuple(kwargs['input_shape'])
      self.batch_input_shape = batch_input_shape
      input_dtype = kwargs.get('input_dtype', tf.float32)
      self.input_dtype = input_dtype

  def add_weight(self, shape, initializer, regularizer=None, name=None):
    """Adds a weight variable to the layer.

    Parameters
    ----------
    shape: The shape tuple of the weight.
    initializer: An Initializer instance (callable).
    regularizer: An optional Regularizer instance.
    """
    initializer = initializations.get(initializer)
    weight = initializer(shape, name=name)
    if regularizer is not None:
      self.add_loss(regularizer(weight))

    return weight

  def add_loss(self, losses, inputs=None):
    """Adds losses to model."""
    if losses is None:
      return
    # Update self.losses
    losses = to_list(losses)
    if not hasattr(self, 'losses'):
      self.losses = []
    try:
      self.losses += losses
    except AttributeError:
      # In case self.losses isn't settable
      # (i.e. it's a getter method).
      # In that case the `losses` property is
      # auto-computed and shouldn't be set.
      pass
    # Update self._per_input_updates
    if not hasattr(self, '_per_input_losses'):
      self._per_input_losses = {}
    if inputs is not None:
      inputs_hash = object_list_uid(inputs)
    else:
      # Updates indexed by None are unconditional
      # rather than input-dependent
      inputs_hash = None
    if inputs_hash not in self._per_input_losses:
      self._per_input_losses[inputs_hash] = []
    self._per_input_losses[inputs_hash] += losses

  def call(self, x):
    """This is where the layer's logic lives.

    Parameters
    ----------
    x: input tensor, or list/tuple of input tensors.

    Returns
    -------
    A tensor or list/tuple of tensors.
    """
    return x

  def __call__(self, x):
    """Wrapper around self.call(), for handling
    Parameters
    ----------
    x: Can be a tensor or list/tuple of tensors.
    """
    return self.call(x)
    #outputs = to_list(self.call(x))
    #return outputs


class InputLayer(Layer):
  """Layer to be used as an entry point into a graph.

  Create its a placeholder tensor (pass arguments `input_shape`
  or `batch_input_shape` as well as `input_dtype`).

  Parameters
  ----------
  input_shape: Shape tuple, not including the batch axis.
  batch_input_shape: Shape tuple, including the batch axis.
  input_dtype: Datatype of the input.
  name: Name of the layer (string).
  """

  def __init__(self,
               input_shape=None,
               batch_input_shape=None,
               input_dtype=None,
               name=None):
    self.uses_learning_phase = False
    self.trainable = False

    if not name:
      prefix = 'input'
      # TODO(rbharath): Keras uses a global var here to maintain
      # unique counts. This seems dangerous. How does tensorflow handle?
      name = prefix + '_' + str(model_ops.get_uid(prefix))
    self.name = name

    if input_shape and batch_input_shape:
      raise ValueError('Only provide the input_shape OR '
                       'batch_input_shape argument to '
                       'InputLayer, not both at the same time.')
    if not batch_input_shape:
      if not input_shape:
        raise ValueError('An Input layer should be passed either '
                         'a `batch_input_shape` or an `input_shape`.')
      else:
        batch_input_shape = (None,) + tuple(input_shape)
    else:
      batch_input_shape = tuple(batch_input_shape)

    if not input_dtype:
      input_dtype = tf.float32

    self.batch_input_shape = batch_input_shape
    self.input_dtype = input_dtype

  def __call__(self):
    self.placeholder = tf.placeholder(
        dtype=self.input_dtype, shape=self.batch_input_shape, name=self.name)
    self.placeholder._uses_learning_phase = False
    return [self.placeholder]


def Input(shape=None, batch_shape=None, name=None, dtype=tf.float32):
  """Input() is used to create a placeholder input

  Parameters
  ----------
  shape: A shape tuple (integer), not including the batch size.
      For instance, `shape=(32,)` indicates that the expected input
      will be batches of 32-dimensional vectors.
  name: An optional name string for the layer.
      Should be unique in a model (do not reuse the same name twice).
      It will be autogenerated if it isn't provided.
  dtype: The data type expected by the input, as a string
      (`float32`, `float64`, `int32`...)

  # TODO(rbharath): Support this type of functional API.
  Example:

  >>> # this is a logistic regression in Keras
  >>> a = dc.nn.Input(shape=(32,))
  >>> b = dc.nn.Dense(16)(a)
  >>> model = dc.nn.FunctionalModel(input=a, output=b)
  """
  # If batch size not specified
  if len(shape) == 1:
    batch_shape = (None,) + tuple(shape)
  input_layer = InputLayer(
      batch_input_shape=batch_shape, name=name, input_dtype=dtype)
  return input_layer


class Dense(Layer):
  """Just your regular densely-connected NN layer.

  TODO(rbharath): Make this functional in deepchem

  Example:

  >>> import deepchem as dc
  >>> # as first layer in a sequential model:
  >>> model = dc.models.Sequential()
  >>> model.add(dc.nn.Input(shape=16))
  >>> model.add(dc.nn.Dense(32))
  >>> # now the model will take as input arrays of shape (*, 16)
  >>> # and output arrays of shape (*, 32)

  >>> # this is equivalent to the above:
  >>> model = dc.models.Sequential()
  >>> model.add(dc.nn.Input(shape=16))
  >>> model.add(dc.nn.Dense(32))

  Parameters
  ----------
  output_dim: int > 0.
  init: name of initialization function for the weights of the layer
  activation: name of activation function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: a(x) = x).
  W_regularizer: (eg. L1 or L2 regularization), applied to the main weights matrix.
  b_regularizer: instance of regularize applied to the bias.
  activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
    applied to the network output.
  bias: whether to include a bias
    (i.e. make the layer affine rather than linear).
  input_dim: dimensionality of the input (integer). This argument
    (or alternatively, the keyword argument `input_shape`)
    is required when using this layer as the first layer in a model.

  # Input shape
    nD tensor with shape: (nb_samples, ..., input_dim).
    The most common situation would be
    a 2D input with shape (nb_samples, input_dim).

  # Output shape
    nD tensor with shape: (nb_samples, ..., output_dim).
    For instance, for a 2D input with shape `(nb_samples, input_dim)`,
    the output would have shape `(nb_samples, output_dim)`.
  """

  def __init__(self,
               output_dim,
               input_dim,
               init='glorot_uniform',
               activation="relu",
               bias=True,
               **kwargs):
    self.init = initializations.get(init)
    self.activation = activations.get(activation)
    self.output_dim = output_dim
    self.input_dim = input_dim

    self.bias = bias

    input_shape = (self.input_dim,)
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(Dense, self).__init__(**kwargs)
    self.input_dim = input_dim

  def __call__(self, x):
    self.W = self.add_weight(
        (self.input_dim, self.output_dim),
        initializer=self.init,
        name='{}_W'.format(self.name))
    self.b = self.add_weight(
        (self.output_dim,), initializer='zero', name='{}_b'.format(self.name))

    output = model_ops.dot(x, self.W)
    if self.bias:
      output += self.b
    return output


class Dropout(Layer):
  """Applies Dropout to the input.

  Dropout consists in randomly setting
  a fraction `p` of input units to 0 at each update during training time,
  which helps prevent overfitting.

  Parameters
  ----------
  p: float between 0 and 1. Fraction of the input units to drop.
  seed: A Python integer to use as random seed.

  # References
      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
  """

  def __init__(self, p, seed=None, **kwargs):
    self.p = p
    self.seed = seed
    if 0. < self.p < 1.:
      self.uses_learning_phase = True
    super(Dropout, self).__init__(**kwargs)

  def call(self, x):
    if 0. < self.p < 1.:

      def dropped_inputs():
        retain_prob = 1 - self.p
        return tf.nn.dropout(x * 1., retain_prob, seed=self.seed)

      x = model_ops.in_train_phase(dropped_inputs, lambda: x)
    return x


class BatchNormalization(Layer):
  """Batch normalization layer (Ioffe and Szegedy, 2014).

  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Parameters
  ----------
  epsilon: small float > 0. Fuzz parameter.
  mode: integer, 0, 1 or 2.
    - 0: feature-wise normalization.
        Each feature map in the input will
        be normalized separately. The axis on which
        to normalize is specified by the `axis` argument.
        During training we use per-batch statistics to normalize
        the data, and during testing we use running averages
        computed during the training phase.
    - 1: sample-wise normalization. This mode assumes a 2D input.
    - 2: feature-wise normalization, like mode 0, but
        using per-batch statistics to normalize the data during both
        testing and training.
  axis: integer, axis along which to normalize in mode 0. For instance,
    if your input tensor has shape (samples, channels, rows, cols),
    set axis to 1 to normalize per feature map (channels axis).
  momentum: momentum in the computation of the
    exponential average of the mean and standard deviation
    of the data, for feature-wise normalization.
  beta_init: name of initialization function for shift parameter, or
    alternatively, TensorFlow function to use for weights initialization.
  gamma_init: name of initialization function for scale parameter, or
    alternatively, TensorFlow function to use for weights initialization.
  gamma_regularizer: instance of WeightRegularizer
    (eg. L1 or L2 regularization), applied to the gamma vector.
  beta_regularizer: instance of WeightRegularizer,
    applied to the beta vector.

  Input shape:
  Arbitrary. Use the keyword argument input_shape
  (tuple of integers, does not include the samples axis)
  when using this layer as the first layer in a model.

  Output shape:
  Same shape as input.

  References:
    - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  def __init__(self,
               epsilon=1e-3,
               mode=0,
               axis=-1,
               momentum=0.99,
               beta_init='zero',
               gamma_init='one',
               gamma_regularizer=None,
               beta_regularizer=None,
               **kwargs):
    self.beta_init = initializations.get(beta_init)
    self.gamma_init = initializations.get(gamma_init)
    self.epsilon = epsilon
    self.mode = mode
    self.axis = axis
    self.momentum = momentum
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    if self.mode == 0:
      self.uses_learning_phase = True
    super(BatchNormalization, self).__init__(**kwargs)

  def build(self, input_shape):
    shape = (input_shape[self.axis],)

    self.gamma = self.add_weight(
        shape,
        initializer=self.gamma_init,
        regularizer=self.gamma_regularizer,
        name='{}_gamma'.format(self.name))
    self.beta = self.add_weight(
        shape,
        initializer=self.beta_init,
        regularizer=self.beta_regularizer,
        name='{}_beta'.format(self.name))
    # Not Trainable
    self.running_mean = self.add_weight(
        shape, initializer='zero', name='{}_running_mean'.format(self.name))
    # Not Trainable
    self.running_std = self.add_weight(
        shape, initializer='one', name='{}_running_std'.format(self.name))

  def call(self, x):
    if not isinstance(x, list):
      input_shape = model_ops.int_shape(x)
    else:
      x = x[0]
      input_shape = model_ops.int_shape(x)
    self.build(input_shape)
    if self.mode == 0 or self.mode == 2:

      reduction_axes = list(range(len(input_shape)))
      del reduction_axes[self.axis]
      broadcast_shape = [1] * len(input_shape)
      broadcast_shape[self.axis] = input_shape[self.axis]

      x_normed, mean, std = model_ops.normalize_batch_in_training(
          x, self.gamma, self.beta, reduction_axes, epsilon=self.epsilon)

      if self.mode == 0:
        self.add_update([
            model_ops.moving_average_update(self.running_mean, mean,
                                            self.momentum),
            model_ops.moving_average_update(self.running_std, std,
                                            self.momentum)
        ], x)

        if sorted(reduction_axes) == range(model_ops.get_ndim(x))[:-1]:
          x_normed_running = tf.nn.batch_normalization(
              x,
              self.running_mean,
              self.running_std,
              self.beta,
              self.gamma,
              epsilon=self.epsilon)
        else:
          # need broadcasting
          broadcast_running_mean = tf.reshape(self.running_mean,
                                              broadcast_shape)
          broadcast_running_std = tf.reshape(self.running_std, broadcast_shape)
          broadcast_beta = tf.reshape(self.beta, broadcast_shape)
          broadcast_gamma = tf.reshape(self.gamma, broadcast_shape)
          x_normed_running = tf.batch_normalization(
              x,
              broadcast_running_mean,
              broadcast_running_std,
              broadcast_beta,
              broadcast_gamma,
              epsilon=self.epsilon)

        # pick the normalized form of x corresponding to the training phase
        x_normed = model_ops.in_train_phase(x_normed, x_normed_running)

    elif self.mode == 1:
      # sample-wise normalization
      m = model_ops.mean(x, axis=-1, keepdims=True)
      std = model_ops.sqrt(
          model_ops.var(x, axis=-1, keepdims=True) + self.epsilon)
      x_normed = (x - m) / (std + self.epsilon)
      x_normed = self.gamma * x_normed + self.beta
    return x_normed
