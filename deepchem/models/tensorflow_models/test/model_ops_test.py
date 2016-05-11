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


import os


import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.training import checkpoint_state_pb2 as cspb

from deepchem.models.tensorflow_models import model_ops

FLAGS = flags.FLAGS
FLAGS.test_random_seed = 20151102

def _numpy_radial_cutoff(R):
  Rc = 6.
  FC = 0.5*(np.cos(np.pi*R/Rc)+1)*(R<=Rc)
  N = R.shape[0]
  for i in range(N):
    FC[i,i] = 0
  return FC

def _numpy_gaussian_distance_matrix(R):
  return np.exp(-1.*R**2)

def _numpy_radial_symmetry_function(R):
  K = _numpy_gaussian_distance_matrix(R)
  FC = _numpy_radial_cutoff(R)
  N = R.shape[0]
  return np.sum(K * FC, axis=1)

#def _numpy_P_matrix(KFC):
#
#  N = KFC.shape[0]
#  P = np.zeros((N, N, N))
#  for i in range(N):
#    P[i] = np.outer(KFC[i], KFC[i]) * KFC
#  return P

def _numpy_P_matrix(KFC):

  N = KFC.shape[0]
  P = np.zeros((N, N, N))
  K1  = np.repeat(KFC[:,:,np.newaxis], N, axis=2)
  K2  = np.repeat(KFC[:,np.newaxis,:], N, axis=1)
  K3  = np.repeat(KFC[np.newaxis,:,:], N, axis=0)
  P = K1*K2*K3
  #for i in range(N):
  #  P[i] = np.outer(KFC[i], KFC[i]) * KFC
  return P

#def _numpy_cos_matrix(R, D):
#
#  zeta = 1
#  N = R.shape[0]
#  lm = np.eye(N)*1e-5
#  R += lm
#  Rbot = np.zeros((N,N,N))
#  Rtop = np.zeros((N,N,N)) 
#  for i in range(N):
#    Rbot[i] = np.outer(R[i],R[i])
#    Rtop[i] = np.dot(D[i],D[i].T)
#
#  C = Rtop/Rbot
#  C = (1+C)**zeta
#  return C

def _numpy_cos_matrix(R, D, zeta=1):

  N = R.shape[0]
  lm = np.eye(N)*1e-5
  R += lm
  Rbot1  = np.repeat(R[:,:,np.newaxis], N, axis=2)
  Rbot2  = np.repeat(R[:,np.newaxis,:], N, axis=1)
  Rbot = Rbot1*Rbot2
  l = np.arange(N)
  Rtop = np.tensordot(D, D, axes=([2], [2]))[l,:,l,:]
  C = Rtop/Rbot
  C = (1+C)**zeta
  return C

# D.shape == (N, N, 3)
#def __numpy_angular_symmetry_function(R, D):
#  N = R.shape[0]
#  zeta = 1.
#  eta = 1.
#  lambd = 1.
#  K = _numpy_gaussian_distance_matrix(R)
#  FC = _numpy_radial_cutoff(R)
#  KFC = K * FC
#  P = _numpy_P_matrix(KFC)
#  GE = _numpy_cos_matrix(R, D) 
#  GE = GE * P
#  G = np.sum(np.sum(GE, axis=2), axis=1)
#  G *= 2**(1-zeta)
#  return G

def _numpy_angular_symmetry_function(R, D, zeta=1):
  N = R.shape[0]
  eta = 1.
  lambd = 1.

  K = _numpy_gaussian_distance_matrix(R)
  FC = _numpy_radial_cutoff(R)
  KFC = K * FC
  #return KFC
  #G = np.zeros(N)

  P = _numpy_P_matrix(KFC)
  C = _numpy_cos_matrix(R, D)
  #for i in range(N):
  #  for j in range(N):
  #    if i == j: continue
  #    for k in range(N):
  #      if i == k: continue
  #      #costheta = (1+np.dot(D[i,j],D[i,k])/(R[i,j]*R[i,k]))**zeta
  #      costheta = C[i,j,k]
  #      G[i] += P[i,j,k]*costheta
  G = np.sum(P*C, axis=(1,2))
  G *= 2**(1-zeta)
  return G
        
def _numpy_unrolled_angular_symmetry_function(R, D):
  N = R.shape[0]
  zeta = 1.
  eta = 1.
  lambd = 1.

  K = _numpy_gaussian_distance_matrix(R)
  FC = _numpy_radial_cutoff(R)
  KFC = K * FC
  G = np.zeros(N)

  for i in range(N):
    for j in range(N):
      if i == j: continue
      for k in range(N):
        if i == k: continue
        costheta = (1+np.dot(D[i,j],D[i,k])/(R[i,j]*R[i,k]))**zeta
        G[i] += KFC[i,j]*KFC[i,k]*KFC[j,k]*costheta
    G[i] *= 2**(1-zeta)
  return G


def _create_D_tensor(d):
  N = d.shape[0]
  D = np.zeros((N,N,3))
  for i in range(N):
    for j in range(N):
      D[i,j] = d[i]-d[j]
  return D

class ModelOpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(ModelOpsTest, self).setUp()
    self.root = '/tmp'

  def testGaussianDistanceMatrix(self):
    with self.test_session() as sess:
      R = tf.constant([[0.0, 1.0], [1.0, 0.0]], shape=[2, 2])
      K_test_T = model_ops.GaussianDistanceMatrix(R)
      sess.run(tf.initialize_all_variables())
      K_test, eta, Rs = sess.run([K_test_T] + tf.trainable_variables()) 
      self.assertAllClose(K_test, [[1., 1/np.e], [1/np.e, 1.]])

  def testRadialCutoff(self):
    
    with self.test_session() as sess:
      R = tf.constant([[0.0, 1.0], [1.0, 0.0]], shape=[2, 2])
      R_large = tf.constant([[0.0, 7.0], [7.0, 0.0]], shape=[2, 2])
      FC_test_T = model_ops.RadialCutoff(R)
      FC_large_test_T = model_ops.RadialCutoff(R_large)
      FC_test, FC_large_test = sess.run([FC_test_T, FC_large_test_T])
      self.assertAllClose(
          FC_test, _numpy_radial_cutoff(np.array([[0.0, 1.0],[1.0, 0.0]])))
      self.assertAllClose(
          FC_large_test, _numpy_radial_cutoff(np.array([[0.0, 7.0],[7.0, 0.0]])))

  def testRadialSymmetryFunction(self):

    with self.test_session() as sess:
      R = tf.constant([[0.0, 1.0], [1.0, 0.0]], shape=[2, 2])
      R_large = tf.constant([[0.0, 7.0], [7.0, 0.0]], shape=[2, 2])
      G_test_T = model_ops.RadialSymmetryFunction(R)
      G_large_test_T = model_ops.RadialSymmetryFunction(R_large)
      sess.run(tf.initialize_all_variables())
      print(len(tf.trainable_variables()))
      G_test, G_large_test, eta1, Rs1, eta2, Rs2 = \
        sess.run([G_test_T, G_large_test_T] +
                 tf.trainable_variables())
      self.assertAllClose(
          G_test, _numpy_radial_symmetry_function(np.array([[0.0, 1.0],[1.0, 0.0]])))
      self.assertAllClose(
          G_large_test, _numpy_radial_symmetry_function(np.array([[0.0, 7.0],[7.0, 0.0]])))

  def testAngularSymmetryFunction(self):

    with self.test_session() as sess:

      R = tf.constant([[0.0, 1.0], [1.0, 0.0]], shape=[2, 2])
      R_large = tf.constant([[0.0, 7.0], [7.0, 0.0]], shape=[2, 2])
      G_test_T = model_ops.AngularSymmetryFunction(R)
      G_large_test_T = model_ops.AngularSymmetryFunction(R_large)
      sess.run(tf.initialize_all_variables())
      G_test, G_large_test, eta1, Rs1, eta2, Rs2 = \
        sess.run([G_test_T, G_large_test_T] +
                 tf.trainable_variables())
      self.assertAllClose(
          G_test, _numpy_angular_symmetry_function(np.array([[0.0, 1.0],[1.0, 0.0]])))
      self.assertAllClose(
          G_large_test, _numpy_angular_symmetry_function(np.array([[0.0, 7.0],[7.0, 0.0]])))

  def testRefAngularSymmetryFunction(self):
    
    R = np.array([[0., 1., np.sqrt(2)],[1., 0., 1.], [np.sqrt(2), 1., 0.]])
    d = np.array([[0., 0., 0.],[0., 1., 0.], [1., 1., 0.]])
    D = _create_D_tensor(d)

    G_unrolled = _numpy_unrolled_angular_symmetry_function(R, D)
    G = _numpy_angular_symmetry_function(R, D)       
    self.assertAllClose(G_unrolled, G)  

  def testCircularProduct(self):
    N = 4
    KFC = np.random.rand(N,N)
    P_numpy = _numpy_P_matrix(KFC)

    with self.test_session() as sess:
      KFC_t = tf.convert_to_tensor(KFC)
      P_t = model_ops.CircularProduct(KFC_t)
      P_tensorflow = sess.run(P_t)

    self.assertAllClose(P_numpy, P_tensorflow)

  def testEye(self):
    Ns = [1, 3, 5, 7]
    for N in Ns:
      I_numpy = np.eye(N)
      with self.test_session() as sess:
        I_t = model_ops.eye(N)
        I_tensorflow = sess.run(I_t)
      np.testing.assert_allclose(I_numpy, I_tensorflow)
      
  def testCosKernel(self):
    N = 4
    R = np.random.rand(N, N)
    D = np.random.rand(N, N, 3)
    C_numpy = _numpy_cos_matrix(R, D)

    with self.test_session() as sess:
      R_t = tf.convert_to_tensor(R, dtype=tf.float32)
      D_t = tf.convert_to_tensor(D, dtype=tf.float32)
      C_t = model_ops.CosKernel(R_t, D_t)
      C_tensorflow = sess.run(C_t)

    print(np.amax(C_numpy - C_tensorflow))
    np.testing.assert_allclose(C_numpy, C_tensorflow, rtol=1e-3)

  def testBasicAngularSymmetryFunction(self):
    N = 4
    R = np.random.rand(N, N)
    D = np.random.rand(N, N, 3)
    zeta = 1
    A_numpy = _numpy_angular_symmetry_function(R, D, zeta)

    with self.test_session() as sess:
      R_t = tf.convert_to_tensor(R, dtype=tf.float32)
      D_t = tf.convert_to_tensor(D, dtype=tf.float32)
      zeta_t = tf.convert_to_tensor(zeta, dtype=tf.float32)
      A_t = model_ops.AngularSymmetryFunction(R_t, D_t, zeta_t)
      sess.run(tf.initialize_all_variables())
      A_tensorflow = sess.run([A_t] + tf.trainable_variables())[0]

    print(np.amax(A_numpy - A_tensorflow))
    np.testing.assert_allclose(A_numpy, A_tensorflow)


  def testAddBias(self):
    with self.test_session() as sess:
      w_t = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], shape=[2, 3])
      w_biased_t = model_ops.AddBias(w_t, init=tf.constant(5.0, shape=[3]))
      sess.run(tf.initialize_all_variables())
      w, w_biased, bias = sess.run([w_t, w_biased_t] + tf.trainable_variables())
      self.assertAllEqual(w, [[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]])
      self.assertAllEqual(w_biased, [[6.0, 7.0, 8.0],
                                     [9.0, 10.0, 11.0]])
      self.assertAllEqual(bias, [5.0, 5.0, 5.0])

  def testFullyConnectedLayer(self):
    with self.test_session() as sess:
      features = np.random.random((128, 100))
      features_t = tf.constant(features, dtype=tf.float32)
      dense_t = model_ops.FullyConnectedLayer(features_t, 50)
      sess.run(tf.initialize_all_variables())
      features, dense, w, b = sess.run(
          [features_t, dense_t] + tf.trainable_variables())
      expected = np.dot(features, w) + b
      self.assertAllClose(dense, expected)

  def testMultitaskLogits(self):
    with self.test_session() as sess:
      num_tasks = 3
      np.random.seed(FLAGS.test_random_seed)
      features = np.random.random((5, 100))
      logits_t = model_ops.MultitaskLogits(
          tf.constant(features,
                      dtype=tf.float32),
          num_tasks)
      sess.run(tf.initialize_all_variables())
      output = sess.run(tf.trainable_variables() + logits_t)
      w = output[0:-3:2]
      b = output[1:-3:2]
      logits = output[-3:]
      for i in range(num_tasks):
        expected = np.dot(features, w[i]) + b[i]
        self.assertAllClose(logits[i], expected, rtol=1e-5, atol=1e-5)

  def GetModel(self, train=True):
    model_ops.set_training(train)

    # dummy variable for testing Restore
    tf.Variable(tf.constant(10.0, shape=[1]), name='v0')

  def _CheckBatchNormalization(self, features, convolution, mean, variance,
                               mask=None):
    model_ops.set_training(True)
    epsilon = 0.001
    with self.test_session() as sess:
      features_t = tf.constant(features, dtype=tf.float32)
      batch_norm_t = model_ops.BatchNormalize(
          features_t, convolution=convolution, epsilon=epsilon, mask=mask)
      sess.run(tf.initialize_all_variables())
      batch_norm, beta, gamma = sess.run(
          [batch_norm_t] + tf.trainable_variables())
      expected = gamma * (features - mean) / np.sqrt(variance + epsilon) + beta
      self.assertAllClose(batch_norm, np.ma.filled(expected, 0),
                          rtol=1e-5, atol=1e-5)

  def CheckBatchNormalization(self, features, convolution):
    if convolution:
      axis = (0, 1, 2)
    else:
      axis = 0
    mean = features.mean(axis=axis)
    variance = features.var(axis=axis)
    self._CheckBatchNormalization(features, convolution, mean, variance)

  def CheckBatchNormalizationWithMask(self, features, convolution, mask):
    # convert features to a masked array
    # masked array must be created with a mask of the same shape as features
    expanded_mask = np.logical_not(
        np.ones_like(features) * np.expand_dims(mask, -1))
    features = np.ma.array(features, mask=expanded_mask)
    if convolution:
      axis = (0, 1, 2)
      # masked arrays don't support mean/variance with tuple for axis
      count = np.logical_not(features.mask).sum(axis=axis)
      mean = features.sum(axis=axis) / count
      variance = np.square(features - mean).sum(axis=axis) / count
    else:
      axis = 0
      mean = features.mean(axis=axis)
      variance = features.var(axis=axis)
    mask_t = tf.constant(mask, dtype=tf.float32)
    self._CheckBatchNormalization(features, convolution, mean, variance,
                                  mask=mask_t)

  def testBatchNormalization(self):
    # no convolution: norm over batch (first axis)
    self.CheckBatchNormalization(
        features=np.random.random((2, 3, 2, 3)), convolution=False)

  def testBatchNormalizationWithConv(self):
    # convolution: norm over first three axes
    self.CheckBatchNormalization(
        features=np.random.random((2, 3, 2, 3)), convolution=True)

  def testBatchNormalizationInference(self):
    # create a simple batch-normalized model
    model_ops.set_training(True)
    epsilon = 0.001
    decay = 0.95
    checkpoint = os.path.join(self.root, 'my-checkpoint')
    with self.test_session() as sess:
      features = np.random.random((2, 3, 2, 3))
      features_t = tf.constant(features, dtype=tf.float32)
      # create variables for beta, gamma, and moving mean and variance
      model_ops.BatchNormalize(
          features_t, convolution=False, epsilon=epsilon, decay=decay)
      sess.run(tf.initialize_all_variables())
      updates = tf.group(*tf.get_default_graph().get_collection('updates'))
      sess.run(updates)  # update moving mean and variance
      expected_mean, expected_variance, _, _ = tf.all_variables()
      expected_mean = expected_mean.eval()
      expected_variance = expected_variance.eval()

      # save a checkpoint
      saver = tf.train.Saver()
      saver.save(sess, checkpoint)

    super(ModelOpsTest, self).setUp()  # reset the default graph

    # check that the moving mean and variance are used for evaluation
    # get a new set of features to verify that the correct mean and var are used
    model_ops.set_training(False)
    with self.test_session() as sess:
      new_features = np.random.random((2, 3, 2, 3))
      new_features_t = tf.constant(new_features, dtype=tf.float32)
      batch_norm_t = model_ops.BatchNormalize(
          new_features_t, convolution=False, epsilon=epsilon, decay=decay)
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint)
      batch_norm, mean, variance, beta, gamma = sess.run(
          [batch_norm_t] + tf.all_variables())
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_variance)
      expected = (gamma * (new_features - mean) /
                  np.sqrt(variance + epsilon) + beta)
      self.assertAllClose(batch_norm, expected)

  def testBatchNormalizationWithMask(self):
    features = np.random.random((2, 3, 2, 3))
    mask = np.asarray(
        [[[1, 0],
          [1, 1],
          [1, 0]],
         [[0, 1],
          [0, 0],
          [0, 1]]],
        dtype=float)
    self.CheckBatchNormalizationWithMask(
        features=features, convolution=False, mask=mask)

  def testBatchNormalizationWithMaskAndConv(self):
    features = np.random.random((2, 3, 2, 3))
    mask = np.asarray(
        [[[1, 0],
          [1, 1],
          [1, 0]],
         [[0, 1],
          [0, 0],
          [0, 1]]],
        dtype=float)
    self.CheckBatchNormalizationWithMask(
        features=features, convolution=True, mask=mask)

  def testSoftmaxN(self):
    features = np.asarray([[[1, 1],
                            [0.1, 0.3]],
                           [[0, 1],
                            [2, 2]]],
                          dtype=float)
    expected = np.asarray([[[0.5, 0.5],
                            [0.45, 0.55]],
                           [[0.27, 0.73],
                            [0.5, 0.5]]],
                          dtype=float)
    with self.test_session() as sess:
      computed = sess.run(
          model_ops.SoftmaxN(tf.constant(features,
                                         dtype=tf.float32)))
    self.assertAllClose(np.around(computed, 2), expected)

  def testSoftmaxNWithNumpy(self):
    features = np.random.random((2, 3, 4))
    expected = np.exp(features) / np.exp(features).sum(axis=-1, keepdims=True)
    with self.test_session() as sess:
      computed = sess.run(
          model_ops.SoftmaxN(tf.constant(features,
                                         dtype=tf.float32)))
      self.assertAllClose(computed, expected)


if __name__ == '__main__':
  googletest.main()
