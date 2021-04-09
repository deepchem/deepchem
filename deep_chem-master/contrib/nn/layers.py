"""Custom Keras Layers.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import numpy as np
import tensorflow as tf
from deepchem.nn import activations
from deepchem.nn import initializations
from deepchem.nn import model_ops


def affine(x, W, b):
  return tf.matmul(x, W) + b


def tf_affine(x, vm, scope):
  W = vm.var(scope, 'W')
  b = vm.var(scope, 'b')

  return tf.matmul(x, W) + b


def cos(x, y):
  denom = (
      model_ops.sqrt(model_ops.sum(tf.square(x)) * model_ops.sum(tf.square(y)))
      + model_ops.epsilon())
  return model_ops.dot(x, tf.transpose(y)) / denom
