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


class RNN(KerasModel):
  """A 3 dimensional recurrent neural network for either regression or classification.

  Based on deepchem/models/cnn.py.

  The network consists of a configurable number of RNN cells.

  """

  def __init__(self,
               n_tasks,
               n_features,
               n_dims,
               bidirectional=True,
               weight_init_stddevs=0.02,
               bias_init_consts=1.0,
               weight_decay_penalty=0.0,
               weight_decay_penalty_type='l2',
               dropouts=0.5,
               activation_fns=tf.nn.relu,
               dense_layer_size=1000,
               layerType='LSTM',
               mode='classification',
               n_classes=2,
               uncertainty=False,
               residual=False,
               padding='valid',
               **kwargs):
    """Create a RNN.

    In addition to the following arguments, this class also accepts
    all the keyword arguments from TensorGraph.
    """

    if n_dims != 3:
      raise ValueError("Only 3-dimensional RNNs supported at this time.")
    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")
    if residual and padding.lower() != 'same':
      raise ValueError(
          "Residual blocks can only be used when padding is 'same'")
    self.n_tasks = n_tasks
    self.n_features = n_features
    self.dims = n_dims
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

    if layerType == 'LSTM':
      RecurrentLayer = layers.LSTM
    elif layerType == 'GRU':
      RecurrentLayer = layers.GRU
    elif layerType == 'SimpleRNN':
      RecurrentLayer = layers.SimpleRNN
    else:
      raise ValueError('layerType must be "LSTM," "GRU," or "SimpleRNN."')

    if bidirectional == True:
      RecurrentLayer = layers.Bidirectional(RecurrentLayer)
    for filters, size, stride, weight_stddev, bias_const, dropout, activation_fn in zip( #TODO REWRITE FOR LOOP CONDITIONS
        layer_filters, kernel_size, strides, weight_init_stddevs,
        bias_init_consts, dropouts, activation_fns):
      layer = prev_layer
      if next_activation is not None:
        layer = Activation(next_activation)(layer)
      layer = RecurrentLayer(
          size,
          return_sequences=True,
          data_format='channels_last',
          use_bias=(bias_init_consts is not None),
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_stddev),
          bias_initializer=tf.constant_initializer(value=bias_const),
          kernel_regularizer=regularizer)(layer)
      if dropout > 0.0:
        layer = SwitchedDropout(rate=dropout)([layer, dropout_switch])
      if residual and prev_filters == filters: #TODO Cleanup
        prev_layer = Lambda(lambda x: x[0] + x[1])([prev_layer, layer])
      else:
        prev_layer = layer
      next_activation = activation_fn
    if next_activation is not None:
      prev_layer = Activation(activation_fn)(prev_layer)
    prev_layer = PoolLayer()(prev_layer) #TODO FIX
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
    super(RNN, self).__init__(model, loss, output_types=output_types, **kwargs)

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
