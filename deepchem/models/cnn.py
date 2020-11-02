import deepchem as dc
import tensorflow as tf
import numpy as np
from deepchem.models import KerasModel
from deepchem.models.layers import SwitchedDropout
from deepchem.metrics import to_one_hot
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, Lambda
import tensorflow.keras.layers as layers
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection


class CNN(KerasModel):
  """A 1, 2, or 3 dimensional convolutional network for either regression or classification.

  The network consists of the following sequence of layers:

  - A configurable number of convolutional layers
  - A global pooling layer (either max pool or average pool)
  - A final dense layer to compute the output

  It optionally can compose the model from pre-activation residual blocks, as
  described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
  convolution layers.  This often leads to easier training, especially when using a
  large number of layers.  Note that residual blocks can only be used when
  successive layers have the same output shape.  Wherever the output shape changes, a
  simple convolution layer will be used even if residual=True.
  """

  def __init__(self,
               n_tasks,
               n_features,
               dims,
               layer_filters=[100],
               kernel_size=5,
               strides=1,
               weight_init_stddevs=0.02,
               bias_init_consts=1.0,
               weight_decay_penalty=0.0,
               weight_decay_penalty_type='l2',
               dropouts=0.5,
               activation_fns=tf.nn.relu,
               dense_layer_size=1000,
               pool_type='max',
               mode='classification',
               n_classes=2,
               uncertainty=False,
               residual=False,
               padding='valid',
               **kwargs):
    """Create a CNN.

    In addition to the following arguments, this class also accepts
    all the keyword arguments from TensorGraph.

    Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: int
      number of features
    dims: int
      the number of dimensions to apply convolutions over (1, 2, or 3)
    layer_filters: list
      the number of output filters for each convolutional layer in the network.
      The length of this list determines the number of layers.
    kernel_size: int, tuple, or list
      a list giving the shape of the convolutional kernel for each layer.  Each
      element may be either an int (use the same kernel width for every dimension)
      or a tuple (the kernel width along each dimension).  Alternatively this may
      be a single int or tuple instead of a list, in which case the same kernel
      shape is used for every layer.
    strides: int, tuple, or list
      a list giving the stride between applications of the  kernel for each layer.
      Each element may be either an int (use the same stride for every dimension)
      or a tuple (the stride along each dimension).  Alternatively this may be a
      single int or tuple instead of a list, in which case the same stride is
      used for every layer.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization
      of each layer.  The length of this list should equal len(layer_filters)+1,
      where the final element corresponds to the dense layer.  Alternatively this
      may be a single value instead of a list, in which case the same value is used
      for every layer.
    bias_init_consts: list or loat
      the value to initialize the biases in each layer to.  The length of this
      list should equal len(layer_filters)+1, where the final element corresponds
      to the dense layer.  Alternatively this may be a single value instead of a
      list, in which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float
      the dropout probablity to use for each layer.  The length of this list should equal len(layer_filters).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    activation_fns: list or object
      the Tensorflow activation function to apply to each layer.  The length of this list should equal
      len(layer_filters).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.
    pool_type: str
      the type of pooling layer to use, either 'max' or 'average'
    mode: str
      Either 'classification' or 'regression'
    n_classes: int
      the number of classes to predict (only used in classification mode)
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    residual: bool
      if True, the model will be composed of pre-activation residual blocks instead
      of a simple stack of convolutional layers.
    padding: str
      the type of padding to use for convolutional layers, either 'valid' or 'same'
    """
    if dims not in (1, 2, 3):
      raise ValueError('Number of dimensions must be 1, 2, or 3')
    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")
    if residual and padding.lower() != 'same':
      raise ValueError(
          "Residual blocks can only be used when padding is 'same'")
    self.n_tasks = n_tasks
    self.n_features = n_features
    self.dims = dims
    self.mode = mode
    self.n_classes = n_classes
    self.uncertainty = uncertainty
    n_layers = len(layer_filters)
    if not isinstance(kernel_size, list):
      kernel_size = [kernel_size] * n_layers
    if not isinstance(strides, SequenceCollection):
      strides = [strides] * n_layers
    if not isinstance(weight_init_stddevs, SequenceCollection):
      weight_init_stddevs = [weight_init_stddevs] * (n_layers + 1)
    if not isinstance(bias_init_consts, SequenceCollection):
      bias_init_consts = [bias_init_consts] * (n_layers + 1)
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers
    if weight_decay_penalty != 0.0:
      if weight_decay_penalty_type == 'l1':
        regularizer = tf.keras.regularizers.l1(weight_decay_penalty)
      else:
        regularizer = tf.keras.regularizers.l2(weight_decay_penalty)
    else:
      regularizer = None
    if uncertainty:
      if mode != "regression":
        raise ValueError("Uncertainty is only supported in regression mode")
      if any(d == 0.0 for d in dropouts):
        raise ValueError(
            'Dropout must be included in every layer to predict uncertainty')

    # Add the input features.

    features = Input(shape=(None,) * dims + (n_features,))
    dropout_switch = Input(shape=tuple())
    prev_layer = features
    prev_filters = n_features
    next_activation = None

    # Add the convolutional layers

    ConvLayer = (layers.Conv1D, layers.Conv2D, layers.Conv3D)[dims - 1]
    if pool_type == 'average':
      PoolLayer = (layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D,
                   layers.GlobalAveragePooling3D)[dims - 1]
    elif pool_type == 'max':
      PoolLayer = (layers.GlobalMaxPool1D, layers.GlobalMaxPool2D,
                   layers.GlobalMaxPool2D)[dims - 1]
    else:
      raise ValueError('pool_type must be either "average" or "max"')
    for filters, size, stride, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_filters, kernel_size, strides, weight_init_stddevs,
        bias_init_consts, dropouts, activation_fns):
      layer = prev_layer
      if next_activation is not None:
        layer = Activation(next_activation)(layer)
      layer = ConvLayer(
          filters,
          size,
          stride,
          padding=padding,
          data_format='channels_last',
          use_bias=(bias_init_consts is not None),
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_stddev),
          bias_initializer=tf.constant_initializer(value=bias_const),
          kernel_regularizer=regularizer)(layer)
      if dropout > 0.0:
        layer = SwitchedDropout(rate=dropout)([layer, dropout_switch])
      if residual and prev_filters == filters:
        prev_layer = Lambda(lambda x: x[0] + x[1])([prev_layer, layer])
      else:
        prev_layer = layer
      prev_filters = filters
      next_activation = activation_fn
    if next_activation is not None:
      prev_layer = Activation(activation_fn)(prev_layer)
    prev_layer = PoolLayer()(prev_layer)
    if mode == 'classification':
      logits = Reshape((n_tasks,
                        n_classes))(Dense(n_tasks * n_classes)(prev_layer))
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss = dc.models.losses.SoftmaxCrossEntropy()
    else:
      output = Reshape((n_tasks,))(Dense(
          n_tasks,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_init_stddevs[-1]),
          bias_initializer=tf.constant_initializer(
              value=bias_init_consts[-1]))(prev_layer))
      if uncertainty:
        log_var = Reshape((n_tasks, 1))(Dense(
            n_tasks,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=weight_init_stddevs[-1]),
            bias_initializer=tf.constant_initializer(value=0.0))(prev_layer))
        var = Activation(tf.exp)(log_var)
        outputs = [output, var, output, log_var]
        output_types = ['prediction', 'variance', 'loss', 'loss']

        def loss(outputs, labels, weights):
          diff = labels[0] - outputs[0]
          return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
      else:
        outputs = [output]
        output_types = ['prediction']
        loss = dc.models.losses.L2Loss()
    model = tf.keras.Model(inputs=[features, dropout_switch], outputs=outputs)
    super(CNN, self).__init__(model, loss, output_types=output_types, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if self.mode == 'classification':
          if y_b is not None:
            y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                -1, self.n_tasks, self.n_classes)
        if mode == 'predict':
          dropout = np.array(0.0)
        else:
          dropout = np.array(1.0)
        yield ([X_b, dropout], [y_b], [w_b])
