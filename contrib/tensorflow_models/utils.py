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
"""Utils for graph convolution models."""

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

from google.protobuf import text_format

from tensorflow.python.platform import gfile
from tensorflow.python.training import checkpoint_state_pb2


def ParseCheckpoint(checkpoint):
  """Parse a checkpoint file.

  Args:
    checkpoint: Path to checkpoint. The checkpoint is either a serialized
      CheckpointState proto or an actual checkpoint file.

  Returns:
    The path to an actual checkpoint file.
  """
  warnings.warn("ParseCheckpoint is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  with open(checkpoint) as f:
    try:
      cp = checkpoint_state_pb2.CheckpointState()
      text_format.Merge(f.read(), cp)
      return cp.model_checkpoint_path
    except text_format.ParseError:
      return checkpoint


def Mask(t, mask):
  """Apply a mask to a tensor.

  If not None, mask should be a t.shape[:-1] tensor of 0,1 values.

  Args:
    t: Input tensor.
    mask: Boolean mask with shape == t.shape[:-1]. If None, nothing happens.

  Returns:
    A tensor with the same shape as the input tensor.

  Raises:
    ValueError: If shapes do not match.
  """
  warnings.warn("Mask is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  if mask is None:
    return t
  if not t.get_shape()[:-1].is_compatible_with(mask.get_shape()):
    raise ValueError('Shapes do not match: %s vs. %s' % (t.get_shape(),
                                                         mask.get_shape()))
  return tf.multiply(t, tf.expand_dims(mask, -1))


def Mean(tensor, reduction_indices=None, mask=None):
  """Compute mean using Sum and Mul for better GPU performance.

  See tf.nn.moments for additional notes on this approach.

  Args:
    tensor: Input tensor.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.
    mask: Mask to apply to tensor.

  Returns:
    A tensor with the same type as the input tensor.
  """
  warnings.warn("Mean is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  return Moment(
      1,
      tensor,
      standardize=False,
      reduction_indices=reduction_indices,
      mask=mask)[0]


def Variance(tensor, reduction_indices=None, mask=None):
  """Compute variance.

  Args:
    tensor: Input tensor.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.
    mask: Mask to apply to tensor.

  Returns:
    A tensor with the same type as the input tensor.
  """
  warnings.warn("Variance is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  return Moment(
      2,
      tensor,
      standardize=False,
      reduction_indices=reduction_indices,
      mask=mask)[1]


def Skewness(tensor, reduction_indices=None):
  """Compute skewness, the third standardized moment.

  Args:
    tensor: Input tensor.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.

  Returns:
    A tensor with the same type as the input tensor.
  """
  warnings.warn("Skewness is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  return Moment(
      3, tensor, standardize=True, reduction_indices=reduction_indices)[1]


def Kurtosis(tensor, reduction_indices=None):
  """Compute kurtosis, the fourth standardized moment minus three.

  Args:
    tensor: Input tensor.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.

  Returns:
    A tensor with the same type as the input tensor.
  """
  warnings.warn("Kurtosis is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  return Moment(
      4, tensor, standardize=True, reduction_indices=reduction_indices)[1] - 3


def Moment(k, tensor, standardize=False, reduction_indices=None, mask=None):
  """Compute the k-th central moment of a tensor, possibly standardized.

  Args:
    k: Which moment to compute. 1 = mean, 2 = variance, etc.
    tensor: Input tensor.
    standardize: If True, returns the standardized moment, i.e. the central
      moment divided by the n-th power of the standard deviation.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.
    mask: Mask to apply to tensor.

  Returns:
    The mean and the requested moment.
  """
  warnings.warn("Moment is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  if reduction_indices is not None:
    reduction_indices = np.atleast_1d(reduction_indices).tolist()

  # get the divisor
  if mask is not None:
    tensor = Mask(tensor, mask)
    ones = tf.constant(1, dtype=tf.float32, shape=tensor.get_shape())
    divisor = tf.reduce_sum(
        Mask(ones, mask), axis=reduction_indices, keep_dims=True)
  elif reduction_indices is None:
    divisor = tf.constant(np.prod(tensor.get_shape().as_list()), tensor.dtype)
  else:
    divisor = 1.0
    for i in range(len(tensor.get_shape())):
      if i in reduction_indices:
        divisor *= tensor.get_shape()[i].value
    divisor = tf.constant(divisor, tensor.dtype)

  # compute the requested central moment
  # note that mean is a raw moment, not a central moment
  mean = tf.div(
      tf.reduce_sum(tensor, axis=reduction_indices, keep_dims=True), divisor)
  delta = tensor - mean
  if mask is not None:
    delta = Mask(delta, mask)
  moment = tf.div(
      tf.reduce_sum(
          math_ops.pow(delta, k), axis=reduction_indices, keep_dims=True),
      divisor)
  moment = tf.squeeze(moment, reduction_indices)
  if standardize:
    moment = tf.multiply(
        moment,
        math_ops.pow(
            tf.rsqrt(Moment(2, tensor, reduction_indices=reduction_indices)[1]),
            k))

  return tf.squeeze(mean, reduction_indices), moment


def StringToOp(string):
  """Get a TensorFlow op from a string.

  Args:
    string: String description of an op, such as 'sum' or 'mean'.

  Returns:
    A TensorFlow op.

  Raises:
    NotImplementedError: If string does not match a supported operation.
  """
  warnings.warn("StringToOp is deprecated. "
                "Will be removed in DeepChem 1.4.", DeprecationWarning)
  # TODO(user): median is not implemented yet in TensorFlow
  op_map = {
      'max': tf.reduce_max,
      'mean': Mean,
      'min': tf.reduce_min,
      'sum': tf.reduce_sum,
      'variance': Variance,
      'skewness': Skewness,
      'kurtosis': Kurtosis,
  }
  try:
    return op_map[string]
  except KeyError:
    raise NotImplementedError('Unrecognized op: %s' % string)
