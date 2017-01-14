"""Ops for regularizers

Code borrowed from Keras.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import warnings
from deepchem.nn import model_ops
from deepchem.nn.activations import get_from_module


class Regularizer(object):

  def __call__(self, x):
    return 0

class EigenvalueRegularizer(Regularizer):
  """Regularizer based on the eignvalues of a weight matrix.

  Only available for tensors of rank 2.

  # Arguments
      k: Float; modulates the amount of regularization to apply.
  """

  def __init__(self, k):
    self.k = k

  def __call__(self, x):
    if model_ops.get_ndim(x) != 2:
      raise ValueError('EigenvalueRegularizer '
                       'is only available for tensors of rank 2.')
    covariance = model_ops.dot(tf.transpose(x), x)
    dim1, dim2 = model_ops.eval(tf.shape(covariance))

    # Power method for approximating the dominant eigenvector:
    power = 9  # Number of iterations of the power method.
    o = model_ops.ones([dim1, 1])  # Initial values for the dominant eigenvector.
    main_eigenvect = model_ops.dot(covariance, o)
    for n in range(power - 1):
      main_eigenvect = model_ops.dot(covariance, main_eigenvect)
    covariance_d = model_ops.dot(covariance, main_eigenvect)

    # The corresponding dominant eigenvalue:
    main_eigenval = (model_ops.dot(tf.transpose(covariance_d), main_eigenvect) /
                     model_ops.dot(tf.transpose(main_eigenvect), main_eigenvect))
    # Multiply by the given regularization gain.
    regularization = (main_eigenval ** 0.5) * self.k
    return model_ops.sum(regularization)


class L1L2Regularizer(Regularizer):
  """Regularizer for L1 and L2 regularization.

  # Arguments
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
  """

  def __init__(self, l1=0., l2=0.):
    self.l1 = model_ops.cast_to_floatx(l1)
    self.l2 = model_ops.cast_to_floatx(l2)

  def __call__(self, x):
    regularization = 0
    if self.l1:
        regularization += model_ops.sum(self.l1 * tf.abs(x))
    if self.l2:
        regularization += model_ops.sum(self.l2 * tf.square(x))
    return regularization


# Aliases.
WeightRegularizer = L1L2Regularizer
ActivityRegularizer = L1L2Regularizer

def l1(l=0.01):
  return L1L2Regularizer(l1=l)

def l2(l=0.01):
  return L1L2Regularizer(l2=l)

def l1l2(l1=0.01, l2=0.01):
  return L1L2Regularizer(l1=l1, l2=l2)

def activity_l1(l=0.01):
  return L1L2Regularizer(l1=l)

def activity_l2(l=0.01):
  return L1L2Regularizer(l2=l)

def activity_l1l2(l1=0.01, l2=0.01):
  return L1L2Regularizer(l1=l1, l2=l2)

def get(identifier, kwargs=None):
  return get_from_module(identifier, globals(), 'regularizer',
                         instantiate=True, kwargs=kwargs)
