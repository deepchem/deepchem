"""Test ops for graph construction."""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import tensorflow as tf

import deepchem as dc
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
FLAGS.test_random_seed = 20151102


class TestModelOps(test_util.TensorFlowTestCase):

  def setUp(self):
    super(TestModelOps, self).setUp()
    self.root = '/tmp'

  def test_add_bias(self):
    with self.test_session() as sess:
      w_t = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], shape=[2, 3])
      w_biased_t = dc.nn.add_bias(w_t, init=tf.constant(5.0, shape=[3]))
      sess.run(tf.global_variables_initializer())
      w, w_biased, bias = sess.run([w_t, w_biased_t] + tf.trainable_variables())
      self.assertAllEqual(w, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      self.assertAllEqual(w_biased, [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
      self.assertAllEqual(bias, [5.0, 5.0, 5.0])

  def test_fully_connected_layer(self):
    with self.test_session() as sess:
      features = np.random.random((128, 100))
      features_t = tf.constant(features, dtype=tf.float32)
      dense_t = dc.nn.fully_connected_layer(features_t, 50)
      sess.run(tf.global_variables_initializer())
      features, dense, w, b = sess.run([features_t, dense_t] +
                                       tf.trainable_variables())
      expected = np.dot(features, w) + b
      self.assertAllClose(dense, expected)

  def test_multitask_logits(self):
    with self.test_session() as sess:
      num_tasks = 3
      np.random.seed(FLAGS.test_random_seed)
      features = np.random.random((5, 100))
      logits_t = dc.nn.multitask_logits(
          tf.constant(features, dtype=tf.float32), num_tasks)
      sess.run(tf.global_variables_initializer())
      output = sess.run(tf.trainable_variables() + logits_t)
      w = output[0:-3:2]
      b = output[1:-3:2]
      logits = output[-3:]
      for i in range(num_tasks):
        expected = np.dot(features, w[i]) + b[i]
        self.assertAllClose(logits[i], expected, rtol=1e-5, atol=1e-5)

  def test_softmax_N(self):
    features = np.asarray([[[1, 1], [0.1, 0.3]], [[0, 1], [2, 2]]], dtype=float)
    expected = np.asarray(
        [[[0.5, 0.5], [0.45, 0.55]], [[0.27, 0.73], [0.5, 0.5]]], dtype=float)
    with self.test_session() as sess:
      computed = sess.run(
          dc.nn.softmax_N(tf.constant(features, dtype=tf.float32)))
    self.assertAllClose(np.around(computed, 2), expected)

  def test_softmax_N_with_numpy(self):
    features = np.random.random((2, 3, 4))
    expected = np.exp(features) / np.exp(features).sum(axis=-1, keepdims=True)
    with self.test_session() as sess:
      computed = sess.run(
          dc.nn.softmax_N(tf.constant(features, dtype=tf.float32)))
      self.assertAllClose(computed, expected)
