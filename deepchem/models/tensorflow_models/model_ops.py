#!/usr/bin/python
#
# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ops for graph construction."""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from google.protobuf import text_format

from tensorflow.python.platform import gfile
from tensorflow.python.platform import logging

from deepchem.models.tensorflow_models import utils as model_utils
import sys
import traceback


def AddBias(tensor, init=None, name=None):
  """Add a bias term to a tensor.

  Args:
    tensor: Variable tensor.
    init: Bias initializer. Defaults to zero.
    name: Name for this op. Defaults to tensor.op.name.

  Returns:
    A biased tensor with the same shape as the input tensor.
  """
  if init is None:
    init = tf.zeros([tensor.get_shape()[-1].value])
  with tf.op_scope([tensor], name, tensor.op.name):
    b = tf.Variable(init, name='b')
    return tf.nn.bias_add(tensor, b)


def BatchNormalize(tensor, convolution, mask=None, epsilon=0.001,
                   scale_after_normalization=True, decay=0.999,
                   global_step=None, name=None):
  """Batch normalization.

  Normalize, scale, and shift the input tensor to reduce covariate shift.

  NOTE(user): For inference, the mean and variance must be set to fixed
  values derived from the entire training set. This is accomplished by using
  moving_mean and moving_variance during evaluation. Be sure that models run
  the ops in updates during training or the moving averages will not be very
  useful!

  Args:
    tensor: Input tensor (must be 4D).
    convolution: If True, perform normalization across rows and columns as
      well as over batch.
    mask: Mask to apply to tensor.
    epsilon: Small float to avoid dividing by zero.
    scale_after_normalization: If True, multiply by gamma. If False, gamma is
      not used. When the next layer is linear (also e.g. ReLU), this can be
      disabled since the scaling can be done by the next layer.
    decay: Float value for moving average decay.
    global_step: Tensor containing global step for accelerating moving averages
      at the beginning of training.
    name: Name for this op. Defaults to 'batch_norm'.

  Returns:
    A new tensor corresponding to the batch normalized input.

  Raises:
    ValueError: If the input tensor is not 4D.
  """
  if len(tensor.get_shape()) != 4:
    raise ValueError('Input tensor must be 4D, not %dD'
                     % len(tensor.get_shape()))
  if convolution:
    axes = [0, 1, 2]
    shape = tensor.get_shape()[3:]
  else:
    axes = [0]
    shape = tensor.get_shape()[1:]
  with tf.op_scope([tensor], None, 'BatchNormalize'):
    if mask is not None:
      mean, variance = model_utils.Moment(
          2, tensor, reduction_indices=axes, mask=mask)
    else:
      mean, variance = tf.nn.moments(tensor, axes)

    # Keep track of moving averages for mean and variance. During eval, use the
    # moving averages from training.
    mean_moving_average = MovingAverage(mean, global_step, decay)
    variance_moving_average = MovingAverage(variance, global_step, decay)
    if not is_training():
      mean = mean_moving_average
      variance = variance_moving_average

    beta = tf.Variable(tf.zeros(shape), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=shape), name='gamma')
    if convolution:
      batch_norm = tf.nn.batch_norm_with_global_normalization(
          tensor, mean, variance, beta, gamma, epsilon,
          scale_after_normalization)
    else:
      batch_norm = (tensor - mean) * tf.rsqrt(variance + epsilon)
      if scale_after_normalization:
        batch_norm *= gamma
      batch_norm += beta
    if mask is not None:
      batch_norm = model_utils.Mask(batch_norm, mask)
    return batch_norm


def MovingAverage(tensor, global_step, decay=0.999):
  """Create a variable that contains the moving average of a tensor.

  Adds a tf.identity and special namescope to ensure the tensor
  is colocated with its Variable on the parameter server.
  See http://g/tensorflow-users/PAAXYLlybNs/xA0z-x1qEwAJ
  and replicated_model.py#NameScopeDevicePicker for context.

  Args:
    tensor: Tensor to calculate moving average of.
    global_step: Variable containing the number of global steps.
    decay: Float for exponential decay of moving average.

  Returns:
    A tf.Variable containing the moving average of the input tensor.
  """
  exponential_moving_average = tf.train.ExponentialMovingAverage(
      decay=decay, num_updates=global_step)

  update_op = exponential_moving_average.apply([tensor])
  tf.get_default_graph().add_to_collection('updates', update_op)
  return exponential_moving_average.average(tensor)


def Dropout(tensor, dropout_prob, training_only=True):
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

  Args:
    tensor: Input tensor.
    dropout_prob: Float giving dropout probability for weights (NOT keep
      probability).
    training_only: Boolean. If True (standard dropout), apply dropout only
      during training. If False, apply dropout during inference as well.

  Returns:
    A tensor with the same shape as the input tensor.
  """
  if not dropout_prob:
    return tensor  # do nothing
  keep_prob = 1.0 - dropout_prob
  if is_training() or not training_only:
    tensor = tf.nn.dropout(tensor, keep_prob)
  return tensor


def FullyConnectedLayer(tensor, size, weight_init=None, bias_init=None,
                        name=None):
  """Fully connected layer.

  Args:
    tensor: Input tensor.
    size: Number of nodes in this layer.
    weight_init: Weight initializer.
    bias_init: Bias initializer.
    name: Name for this op. Defaults to 'fully_connected'.

  Returns:
    A new tensor representing the output of the fully connected layer.

  Raises:
    ValueError: If input tensor is not 2D.
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
    w = tf.Variable(weight_init, name='w')
    b = tf.Variable(bias_init, name='b')
    return tf.nn.xw_plus_b(tensor, w, b)

def is_training():
  """Determine whether the default graph is in training mode.

  Returns:
    A boolean value indicating whether the default graph is in training mode.

  Raises:
    ValueError: If the 'train' collection in the default graph does not contain
      exactly one element.
  """
  #traceback.print_stack(file=sys.stdout) 
  train = tf.get_collection("train")
  print("is_training()")
  print("train")
  print(train)
  if not train:
    raise ValueError('Training mode is not set. Please call set_training.')
  elif len(train) > 1:
    raise ValueError('Training mode has more than one setting.')
  return train[0]


def set_training(train):
  """Set the training mode of the default graph.

  This operation may only be called once for a given graph.

  Args:
    graph: Tensorflow graph. 
    train: If True, graph is in training mode.

  Raises:
    AssertionError: If the default graph already has this value set.
  """
  if tf.get_collection('train'):
    raise AssertionError('Training mode already set: %s' %
                         graph.get_collection('train'))
  tf.add_to_collection('train', train)


def MultitaskLogits(features, num_tasks, num_classes=2, weight_init=None,
                    bias_init=None, dropout=None, name=None):
  """Create a logit tensor for each classification task.

  Args:
    features: A 2D tensor with dimensions batch_size x num_features.
    num_tasks: Number of classification tasks.
    num_classes: Number of classes for each task.
    weight_init: Weight initializer.
    bias_init: Bias initializer.
    dropout: Float giving dropout probability for weights (NOT keep
      probability).
    name: Name for this op. Defaults to 'multitask_logits'.

  Returns:
    A list of logit tensors; one for each classification task.
  """
  logits = []
  with tf.name_scope('multitask_logits'):
    for task_idx in range(num_tasks):
      with tf.op_scope([features], name,
                       ('task' + str(task_idx).zfill(len(str(num_tasks))))):
        logits.append(
            Logits(features, num_classes, weight_init=weight_init,
                   bias_init=bias_init, dropout=dropout))
  return logits


def Logits(features, num_classes=2, weight_init=None, bias_init=None,
           dropout=None, name=None):
  """Create a logits tensor for a single classification task.

  You almost certainly don't want dropout on there -- it's like randomly setting
  the (unscaled) probability of a target class to 0.5.

  Args:
    features: A 2D tensor with dimensions batch_size x num_features.
    num_classes: Number of classes for each task.
    weight_init: Weight initializer.
    bias_init: Bias initializer.
    dropout: Float giving dropout probability for weights (NOT keep
      probability).
    name: Name for this op.

  Returns:
    A logits tensor with shape batch_size x num_classes.
  """
  with tf.op_scope([features], name, 'logits') as name:
    return Dropout(
        FullyConnectedLayer(features, num_classes, weight_init=weight_init,
                            bias_init=bias_init, name=name),
        dropout)


def SoftmaxN(tensor, name=None):
  """Apply softmax across last dimension of a tensor.

  Args:
    tensor: Input tensor.
    name: Name for this op. If None, defaults to 'SoftmaxN'.

  Returns:
    A tensor with softmax-normalized values on the last dimension.
  """
  with tf.op_scope([tensor], name, 'SoftmaxN'):
    exp_tensor = tf.exp(tensor)
    reduction_indices = [tensor.get_shape().ndims - 1]
    return tf.div(exp_tensor,
                  tf.reduce_sum(exp_tensor,
                                reduction_indices=reduction_indices,
                                keep_dims=True))


def Transform(tensor, transform, convolution=True, mask=None):
  """Apply a transform to a tensor.

  Args:
    tensor: Input tensor.
    transform: String description of transform. Supported values are 'bias'
      and 'batch_norm'.
    convolution: If True, assume tensor is the output of a convolution.
    mask: Mask to apply to tensor.

  Returns:
    A tensor with the same shape as the input tensor.

  Raises:
    ValueError: If the input tensor is not 3D or 4D.
  """
  if len(tensor.get_shape()) not in [2, 3, 4]:
    raise ValueError('Input tensor must be 2D, 3D or 4D, not %dD.'
                     % len(tensor.get_shape()))
  with tensor.graph.as_default():
    if transform == 'batch_norm':
      # batch normalization requires 4D input
      if len(tensor.get_shape()) != 4:
        # 3D case: add one extra dimension
        if len(tensor.get_shape()) == 3:
          squeeze = [2]
          tensor = tf.expand_dims(tensor, 2)
          if mask is not None:
            mask = tf.expand_dims(mask, -1)
        # 2D case: add two extra dimensions
        else:
          squeeze = [1, 2]
          tensor = tf.expand_dims(tf.expand_dims(tensor, -2), -2)
          if mask is not None:
            mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
        tensor = BatchNormalize(tensor, convolution=convolution, mask=mask)
        tensor = tf.squeeze(tensor, squeeze)
      else:
        tensor = BatchNormalize(tensor, convolution=convolution, mask=mask)
    elif transform == 'bias':
      tensor = AddBias(tensor, init=tf.constant(
          1.0, shape=[tensor.get_shape()[-1].value]))
      if mask is not None:
        tensor = model_utils.Mask(tensor, mask)
  return tensor
