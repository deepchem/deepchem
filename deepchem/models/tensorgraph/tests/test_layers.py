import unittest

import numpy as np
import os
import tensorflow as tf
from nose.tools import assert_true
from tensorflow.python.framework import test_util
from deepchem.models.tensorgraph.layers import Conv1DLayer
from deepchem.models.tensorgraph.layers import Dense

import deepchem as dc

class TestLayers(test_util.TensorFlowTestCase):
  """
  Test that layers function as intended.
  """

  def test_conv_1D_layer(self):
    """Test that Conv1D can be invoked."""
    width = 5
    in_channels = 2
    out_channels = 3
    batch_size = 10
    in_tensor = np.random.rand(batch_size, width, in_channels)
    with self.test_session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Conv1DLayer(width, out_channels)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()

      assert out_tensor.shape == (batch_size, width, out_channels)
    
  def test_dense(self):
    """Test that Dense can be invoked."""
    in_dim = 2
    out_dim = 3
    batch_size = 10
    in_tensor = np.random.rand(batch_size, in_dim)
    with self.test_session() as sess:
      in_tensor = tf.convert_to_tensor(in_tensor, dtype=tf.float32)
      out_tensor = Dense(out_dim)(in_tensor)
      sess.run(tf.global_variables_initializer())
      out_tensor = out_tensor.eval()

      assert out_tensor.shape == (batch_size, out_dim)

  def test_flatten(self):
    """Test that Flatten can be invoked."""
    pass
