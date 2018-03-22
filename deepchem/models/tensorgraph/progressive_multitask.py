from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf
import collections

from deepchem.utils.save import log
from deepchem.metrics import to_one_hot
from deepchem.metrics import from_one_hot
from deepchem.models.tensorgraph.tensor_graph import TensorGraph, TFWrapper
from deepchem.models.tensorgraph.layers import Layer, Feature, Label, Weights, \
    WeightedError, Dense, Dropout, WeightDecay, Reshape, SparseSoftMaxCrossEntropy, \
    L2Loss, ReduceSum, Concat, Stack, TensorWrapper, ReLU, Squeeze, SoftMax, Cast
from deepchem.models.tensorgraph.IRV import Slice


class ProgressiveMultitaskRegressor(TensorGraph):
  """Implements a progressive multitask neural network for regression.
  
  Progressive Networks: https://arxiv.org/pdf/1606.04671v3.pdf

  Progressive networks allow for multitask learning where each task
  gets a new column of weights. As a result, there is no exponential
  forgetting where previous tasks are ignored.

  """

  def __init__(self,
               n_tasks,
               n_features,
               alpha_init_stddevs=0.02,
               layer_sizes=[1000],
               weight_init_stddevs=0.02,
               bias_init_consts=1.0,
               weight_decay_penalty=0.0,
               weight_decay_penalty_type="l2",
               dropouts=0.5,
               activation_fns=tf.nn.relu,
               n_outputs=1,
               **kwargs):
    """Creates a progressive network.
  
    Only listing parameters specific to progressive networks here.

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_features: int
      Number of input features
    alpha_init_stddevs: list
      List of standard-deviations for alpha in adapter layers.
    layer_sizes: list
      the size of each dense layer in the network.  The length of this list determines the number of layers.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization of each layer.  The length
      of this list should equal len(layer_sizes)+1.  The final element corresponds to the output layer.
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    bias_init_consts: list or float
      the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes)+1.
      The final element corresponds to the output layer.  Alternatively this may be a single value instead of a list,
      in which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float
      the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    activation_fns: list or object
      the Tensorflow activation function to apply to each layer.  The length of this list should equal
      len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.
    """

    super(ProgressiveMultitaskRegressor, self).__init__(**kwargs)
    self.n_tasks = n_tasks
    self.n_features = n_features
    self.layer_sizes = layer_sizes
    self.alpha_init_stddevs = alpha_init_stddevs
    self.weight_init_stddevs = weight_init_stddevs
    self.bias_init_consts = bias_init_consts
    self.dropouts = dropouts
    self.activation_fns = activation_fns
    self.n_outputs = n_outputs

    n_layers = len(layer_sizes)
    if not isinstance(weight_init_stddevs, collections.Sequence):
      self.weight_init_stddevs = [weight_init_stddevs] * n_layers
    if not isinstance(alpha_init_stddevs, collections.Sequence):
      self.alpha_init_stddevs = [alpha_init_stddevs] * n_layers
    if not isinstance(bias_init_consts, collections.Sequence):
      self.bias_init_consts = [bias_init_consts] * n_layers
    if not isinstance(dropouts, collections.Sequence):
      self.dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, collections.Sequence):
      self.activation_fns = [activation_fns] * n_layers

    # Add the input features.
    self.mol_features = Feature(shape=(None, n_features))
    self._task_labels = Label(shape=(None, n_tasks))
    self._task_weights = Weights(shape=(None, n_tasks))

    all_layers = {}
    outputs = []
    for task in range(self.n_tasks):
      task_layers = []
      for i in range(n_layers):
        if i == 0:
          prev_layer = self.mol_features
        else:
          prev_layer = all_layers[(i - 1, task)]
          if task > 0:
            lateral_contrib, trainables = self.add_adapter(all_layers, task, i)
            task_layers.extend(trainables)

        layer = Dense(
            in_layers=[prev_layer],
            out_channels=layer_sizes[i],
            activation_fn=None,
            weights_initializer=TFWrapper(
                tf.truncated_normal_initializer,
                stddev=self.weight_init_stddevs[i]),
            biases_initializer=TFWrapper(
                tf.constant_initializer, value=self.bias_init_consts[i]))
        task_layers.append(layer)

        if i > 0 and task > 0:
          layer = layer + lateral_contrib
        assert self.activation_fns[i] is tf.nn.relu, "Only ReLU is supported"
        layer = ReLU(in_layers=[layer])
        if self.dropouts[i] > 0.0:
          layer = Dropout(self.dropouts[i], in_layers=[layer])
        all_layers[(i, task)] = layer

      prev_layer = all_layers[(n_layers - 1, task)]
      layer = Dense(
          in_layers=[prev_layer],
          out_channels=n_outputs,
          weights_initializer=TFWrapper(
              tf.truncated_normal_initializer,
              stddev=self.weight_init_stddevs[-1]),
          biases_initializer=TFWrapper(
              tf.constant_initializer, value=self.bias_init_consts[-1]))
      task_layers.append(layer)

      if task > 0:
        lateral_contrib, trainables = self.add_adapter(all_layers, task,
                                                       n_layers)
        task_layers.extend(trainables)
        layer = layer + lateral_contrib
      output_layer = self.create_output(layer)
      outputs.append(output_layer)

      label = Slice(task, axis=1, in_layers=[self._task_labels])
      weight = Slice(task, axis=1, in_layers=[self._task_weights])
      task_loss = self.create_loss(layer, label, weight)
      self.create_submodel(layers=task_layers, loss=task_loss, optimizer=None)

    outputs = Stack(axis=1, in_layers=outputs)
    self.add_output(outputs)

    # Weight decay not activated
    """
    if weight_decay_penalty != 0.0:
      weighted_loss = WeightDecay(
          weight_decay_penalty,
          weight_decay_penalty_type,
          in_layers=[weighted_loss])
    """

  def create_loss(self, layer, label, weight):
    weighted_loss = ReduceSum(L2Loss(in_layers=[label, layer, weight]))
    return weighted_loss

  def create_output(self, layer):
    return layer

  def add_adapter(self, all_layers, task, layer_num):
    """Add an adapter connection for given task/layer combo"""
    i = layer_num
    prev_layers = []
    trainable_layers = []
    # Handle output layer
    if i < len(self.layer_sizes):
      layer_sizes = self.layer_sizes
      alpha_init_stddev = self.alpha_init_stddevs[i]
      weight_init_stddev = self.weight_init_stddevs[i]
      bias_init_const = self.bias_init_consts[i]
    elif i == len(self.layer_sizes):
      layer_sizes = self.layer_sizes + [self.n_outputs]
      alpha_init_stddev = self.alpha_init_stddevs[-1]
      weight_init_stddev = self.weight_init_stddevs[-1]
      bias_init_const = self.bias_init_consts[-1]
    else:
      raise ValueError("layer_num too large for add_adapter.")
    # Iterate over all previous tasks.
    for prev_task in range(task):
      prev_layers.append(all_layers[(i - 1, prev_task)])
    # prev_layers is a list with elements of size
    # (batch_size, layer_sizes[i-1])
    prev_layer = Concat(axis=1, in_layers=prev_layers)
    with self._get_tf("Graph").as_default():
      alpha = TensorWrapper(
          tf.Variable(
              tf.truncated_normal((1,), stddev=alpha_init_stddev),
              name="alpha_layer_%d_task%d" % (i, task)))
      trainable_layers.append(alpha)

    prev_layer = prev_layer * alpha
    dense1 = Dense(
        in_layers=[prev_layer],
        out_channels=layer_sizes[i - 1],
        activation_fn=None,
        weights_initializer=TFWrapper(
            tf.truncated_normal_initializer, stddev=weight_init_stddev),
        biases_initializer=TFWrapper(
            tf.constant_initializer, value=bias_init_const))
    trainable_layers.append(dense1)

    dense2 = Dense(
        in_layers=[dense1],
        out_channels=layer_sizes[i],
        activation_fn=None,
        weights_initializer=TFWrapper(
            tf.truncated_normal_initializer, stddev=weight_init_stddev),
        biases_initializer=None)
    trainable_layers.append(dense2)

    return dense2, trainable_layers

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          checkpoint_interval=1000,
          deterministic=False,
          restore=False,
          **kwargs):
    for task in range(self.n_tasks):
      self.fit_task(
          dataset,
          nb_epoch=nb_epoch,
          max_checkpoints_to_keep=max_checkpoints_to_keep,
          checkpoint_interval=checkpoint_interval,
          deterministic=deterministic,
          restore=restore,
          submodel=task,
          **kwargs)

  def fit_task(self,
               dataset,
               nb_epoch=10,
               max_checkpoints_to_keep=5,
               checkpoint_interval=1000,
               deterministic=False,
               restore=False,
               submodel=None,
               **kwargs):
    """Fit one task."""
    generator = self.default_generator(
        dataset, epochs=nb_epoch, deterministic=deterministic)
    self.fit_generator(generator, max_checkpoints_to_keep, checkpoint_interval,
                       restore, self.submodels[submodel])


class ProgressiveMultitaskClassifier(ProgressiveMultitaskRegressor):
  """Implements a progressive multitask neural network for classification.
  
  Progressive Networks: https://arxiv.org/pdf/1606.04671v3.pdf

  Progressive networks allow for multitask learning where each task
  gets a new column of weights. As a result, there is no exponential
  forgetting where previous tasks are ignored.

  """

  def __init__(self,
               n_tasks,
               n_features,
               alpha_init_stddevs=0.02,
               layer_sizes=[1000],
               weight_init_stddevs=0.02,
               bias_init_consts=1.0,
               weight_decay_penalty=0.0,
               weight_decay_penalty_type="l2",
               dropouts=0.5,
               activation_fns=tf.nn.relu,
               **kwargs):
    n_outputs = 2
    super(ProgressiveMultitaskClassifier, self).__init__(
        n_tasks,
        n_features,
        alpha_init_stddevs=alpha_init_stddevs,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        weight_decay_penalty=weight_decay_penalty,
        weight_decay_penalty_type=weight_decay_penalty_type,
        dropouts=dropouts,
        activation_fns=activation_fns,
        n_outputs=n_outputs,
        **kwargs)

  def create_loss(self, layer, label, weight):
    task_label = Squeeze(squeeze_dims=1, in_layers=[label])
    task_label = Cast(dtype=tf.int32, in_layers=[task_label])
    task_weight = Squeeze(squeeze_dims=1, in_layers=[weight])

    loss = SparseSoftMaxCrossEntropy(in_layers=[task_label, layer])
    weighted_loss = WeightedError(in_layers=[loss, task_weight])
    return weighted_loss

  def create_output(self, layer):
    output = SoftMax(in_layers=[layer])
    return output
