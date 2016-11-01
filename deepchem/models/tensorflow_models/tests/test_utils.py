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

import tempfile


import numpy as np
import scipy.stats
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_state_pb2

from deepchem.models.tensorflow_models import utils

FLAGS = flags.FLAGS
FLAGS.test_random_seed = 20151102


class UtilsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    np.random.seed(FLAGS.test_random_seed)

  def testParseCheckpoint(self):
    # parse CheckpointState proto
    with tempfile.NamedTemporaryFile(mode='w+') as f:
      cp = checkpoint_state_pb2.CheckpointState()
      cp.model_checkpoint_path = 'my-checkpoint'
      f.write(text_format.MessageToString(cp))
      f.file.flush()
      self.assertEqual(utils.ParseCheckpoint(f.name), 'my-checkpoint')
    # parse path to actual checkpoint
    with tempfile.NamedTemporaryFile(mode='w+') as f:
      f.write('This is not a CheckpointState proto.')
      f.file.flush()
      self.assertEqual(utils.ParseCheckpoint(f.name), f.name)

  def PrepareFeatures(self, features):
    features = np.asarray(features, dtype=float)
    features_t = tf.constant(features, dtype=tf.float32)
    return features, features_t

  def PrepareMask(self, features, mask):
    mask = np.asarray(mask, dtype=float)
    mask_t = tf.constant(mask, dtype=tf.float32)
    # the provided mask has to be the same shape as features
    expanded_mask = np.logical_not(
        np.ones_like(features) * np.expand_dims(mask, -1))
    masked_features = np.ma.masked_array(features, mask=expanded_mask)
    return masked_features, mask_t

  def Check(self, func, features, expected, axis=None, mask=None):
    with self.test_session() as sess:
      features, features_t = self.PrepareFeatures(features)
      if mask is not None:
        features, mask = self.PrepareMask(features, mask)
      self.assertAllClose(
          sess.run(func(features_t, reduction_indices=axis, mask=mask)),
          expected)

  def testMean(self):
    self.Check(utils.Mean,
               features=[0, 1],
               expected=0.5)
    self.Check(utils.Mean,
               features=[[0, 1],
                         [2, 3]],
               expected=[0.5, 2.5],
               axis=1)
    self.Check(utils.Mean,
               features=[[[0, 1],
                          [2, 3]],
                         [[4, 5],
                          [6, 7]]],
               expected=[2.5, 4.5],
               axis=[0, 2])

  def testMeanWithMask(self):
    self.Check(utils.Mean,
               features=[[9999],
                         [1],
                         [2]],
               expected=1.5,
               mask=[0, 1, 1])
    self.Check(utils.Mean,
               features=[[0, 1],
                         [9999, 9999]],
               expected=[0, 1],
               axis=0,
               mask=[1, 0])
    self.Check(utils.Mean,
               features=[[[0, 1],
                          [9999, 9999]],
                         [[9999, 9999],
                          [6, 7]]],
               expected=[0.5, 6.5],
               axis=[0, 2],
               mask=[[1, 0],
                     [0, 1]])

  def testVariance(self):
    self.Check(utils.Variance,
               features=[0, 1],
               expected=0.25)
    self.Check(utils.Variance,
               features=[[0, 2],
                         [2, 3]],
               expected=[1, 0.25],
               axis=1)
    self.Check(utils.Variance,
               features=[[[0, 1],
                          [2, 3]],
                         [[4, 5],
                          [6, 7]]],
               expected=[4.25, 4.25],
               axis=[0, 2])

  def testVarianceWithMask(self):
    self.Check(utils.Variance,
               features=[[0],
                         [1],
                         [2]],
               expected=0.25,
               mask=[0, 1, 1])
    self.Check(utils.Variance,
               features=[[0, 2],
                         [9999, 9999],
                         [4, 4]],
               expected=[4, 1],
               axis=0,
               mask=[1, 0, 1])
    self.Check(utils.Variance,
               features=[[[0, 1],
                          [9999, 9999]],
                         [[9999, 9999],
                          [6, 8]]],
               expected=[0.25, 1],
               axis=[0, 2],
               mask=[[1, 0],
                     [0, 1]])

  def testMoment(self):
    with self.test_session() as sess:
      features = np.random.random((3, 4, 5))
      features_t = tf.constant(features, dtype=tf.float32)

      # test k = 1..4
      for k in [1, 2, 3, 4]:
        # central moments
        self.assertAllClose(
            sess.run(utils.Moment(k, features_t)[1]),
            scipy.stats.moment(features, k, axis=None),
            rtol=1e-5, atol=1e-5)

        # standardized moments
        self.assertAllClose(
            sess.run(utils.Moment(k, features_t, standardize=True)[1]),
            np.divide(scipy.stats.moment(features, k, axis=None),
                      np.power(features.std(), k)),
            rtol=1e-5, atol=1e-5)

        # central across one axis
        self.assertAllClose(
            sess.run(utils.Moment(k, features_t, reduction_indices=1)[1]),
            scipy.stats.moment(features, k, axis=1),
            rtol=1e-5, atol=1e-5)

        # standardized across one axis
        self.assertAllClose(
            sess.run(utils.Moment(k, features_t, standardize=True,
                                  reduction_indices=1)[1]),
            np.divide(scipy.stats.moment(features, k, axis=1),
                      np.power(features.std(axis=1), k)),
            rtol=1e-5, atol=1e-5)

  def testSkewness(self):
    with self.test_session() as sess:
      features = np.random.random((3, 4, 5))
      features_t = tf.constant(features, dtype=tf.float32)
      self.assertAllClose(sess.run(utils.Skewness(features_t)),
                          scipy.stats.skew(features, axis=None),
                          rtol=1e-5, atol=1e-5)
      self.assertAllClose(sess.run(utils.Skewness(features_t, 1)),
                          scipy.stats.skew(features, axis=1),
                          rtol=1e-5, atol=1e-5)

  def testKurtosis(self):
    with self.test_session() as sess:
      features = np.random.random((3, 4, 5))
      features_t = tf.constant(features, dtype=tf.float32)
      self.assertAllClose(sess.run(utils.Kurtosis(features_t)),
                          scipy.stats.kurtosis(features, axis=None),
                          rtol=1e-5, atol=1e-5)
      self.assertAllClose(sess.run(utils.Kurtosis(features_t, 1)),
                          scipy.stats.kurtosis(features, axis=1),
                          rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
  googletest.main()
