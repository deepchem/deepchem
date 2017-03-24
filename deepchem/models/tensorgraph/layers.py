import random
import string

import tensorflow as tf


class Layer(object):
  def __init__(self, **kwargs):
    if "name" not in kwargs:
      self.name = self._random_name()
    else:
      self.name = kwargs['name']

  def _random_name(self):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))


class Conv1DLayer(Layer):
  def __init__(self, width, out_channels, **kwargs):
    self.width = width
    self.out_channels = out_channels
    self.out_tensor = None
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 1:
      raise ValueError("Only One Parent to conv1D over")
    parent = parents[0]
    if len(parent.out_tensor.get_shape()) != 3:
      raise ValueError("Parent tensor must be (batch, width, channel)")
    parent_shape = parent.out_tensor.get_shape()
    parent_channel_size = parent_shape[2].value
    with tf.name_scope(self.name):
      f = tf.Variable(
        tf.random_normal([self.width, parent_channel_size, self.out_channels
                          ]))
      b = tf.Variable(tf.random_normal([self.out_channels]))
      t = tf.nn.conv1d(parent.out_tensor, f, stride=1, padding="SAME")
      t = tf.nn.bias_add(t, b)
      self.out_tensor = tf.nn.relu(t)


class Dense(Layer):
  def __init__(self, out_channels, activation_fn=tf.nn.sigmoid, **kwargs):
    self.out_channels = out_channels
    self.out_tensor = None
    self.activation_fn = activation_fn
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 1:
      raise ValueError("Only One Parent to Dense over")
    parent = parents[0]
    if len(parent.out_tensor.get_shape()) != 2:
      raise ValueError("Parent tensor must be (batch, width)")
    self.out_tensor = tf.contrib.layers.fully_connected(parent.out_tensor,
                                                        num_outputs=self.out_channels,
                                                        activation_fn=self.activation_fn,
                                                        scope=self.name,
                                                        trainable=True)
    return self.out_tensor


class Flatten(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 1:
      raise ValueError("Only One Parent to conv1D over")
    parent = parents[0]
    if len(parent.out_tensor.get_shape()) != 3:
      raise ValueError("Parent tensor must be (batch, width, channel)")
    parent_shape = parent.out_tensor.get_shape()
    vector_size = parent_shape[1].value * parent_shape[2].value
    parent_tensor = parent.out_tensor
    with tf.name_scope(self.name):
      self.out_tensor = tf.reshape(parent_tensor, shape=(-1, vector_size))
    return self.out_tensor


class CombineMeanStd(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 2:
      raise ValueError("Must have two parents")
    mean_parent, std_parent = parents[0], parents[1]
    mean_parent_tensor, std_parent_tensor = mean_parent.out_tensor, std_parent.out_tensor
    with tf.name_scope(self.name):
      sample_noise = tf.random_normal(
        mean_parent_tensor.get_shape(), 0, 1, dtype=tf.float32)
      self.out_tensor = mean_parent_tensor + (std_parent_tensor * sample_noise)


class Repeat(Layer):
  def __init__(self, n_times, **kwargs):
    self.n_times = n_times
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = parents[0].out_tensor
    with tf.name_scope(self.name):
      t = tf.expand_dims(parent_tensor, 1)
      pattern = tf.stack([1, self.n_times, 1])
      self.out_tensor = tf.tile(t, pattern)


class GRU(Layer):
  def __init__(self, n_hidden, out_channels, batch_size, **kwargs):
    self.n_hidden = n_hidden
    self.out_channels = out_channels
    self.batch_size = batch_size
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = parents[0].out_tensor
    with tf.name_scope(self.name):
      gru_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
      initial_gru_state = gru_cell.zero_state(self.batch_size, tf.float32)
      rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        gru_cell,
        parent_tensor,
        initial_state=initial_gru_state,
        scope=self.name)
      projection = lambda x: tf.contrib.layers.linear(x, num_outputs=self.out_channels, activation_fn=tf.nn.sigmoid)
      self.out_tensor = tf.map_fn(projection, rnn_outputs)


class TimeSeriesDense(Layer):
  def __init__(self, out_channels, **kwargs):
    self.out_channels = out_channels
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = parents[0].out_tensor
    with tf.name_scope(self.name):
      dense_fn = lambda x: tf.contrib.layers.fully_connected(x, num_outputs=self.out_channels,
                                                             activation_fn=tf.nn.sigmoid)
      self.out_tensor = tf.map_fn(dense_fn, parent_tensor)


class Input(Layer):
  def __init__(self, t_shape, **kwargs):
    self.t_shape = t_shape
    super().__init__(**kwargs)

  def __call__(self, *parents):
    with tf.name_scope(self.name):
      self.out_tensor = tf.placeholder(tf.float32, shape=self.t_shape)


class LossLayer(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, *parents):
    guess, label = parents[0], parents[1]
    self.out_tensor = tf.reduce_mean(tf.square(guess.out_tensor - label.out_tensor))
    return self.out_tensor


class SoftMax(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 1:
      raise ValueError("Must only Softmax single parent")
    parent = parents[0]
    self.out_tensor = tf.contrib.layers.softmax(parent.out_tensor)
    return self.out_tensor


class Concat(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) <= 1:
      raise ValueError("Concat must join at least two tensors")
    out_tensors = [x.out_tensor for x in parents]

    self.out_tensor = tf.concat(out_tensors, 1)
    return self.out_tensor


class SoftMaxCrossEntropy(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, *parents):
    if len(parents) != 2:
      raise ValueError()
    labels, logits = parents[0].out_tensor, parents[1].out_tensor
    self.out_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=labels, labels=logits)
    self.out_tensor = tf.reshape(self.out_tensor, [-1, 1])
    return self.out_tensor
