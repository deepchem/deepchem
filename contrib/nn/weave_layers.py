#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:02:04 2017

@author: michael
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from deepchem.nn import activations
from deepchem.nn import initializations
from deepchem.nn import model_ops

# TODO(rbharath): This class does not yet have a
# TensorGraph equivalent, but one may not be required.
# Commented out for now, remove if OK.
class AlternateWeaveLayer(WeaveLayer):
  """ Alternate implementation of weave module
      same variables, different graph structures
  """

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [atom_features, pair_features, pair_split, atom_split, atom_to_pair]

    Parameters
    ----------
    x: list
      list of Tensors of form described above.
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    A: Tensor
      Tensor of atom_features
    P: Tensor
      Tensor of pair_features
    """
    # Add trainable weights
    self.build()

    atom_features = x[0]
    pair_features = x[1]

    pair_split = x[2]
    atom_to_pair = x[4]

    AA = tf.matmul(atom_features, self.W_AA) + self.b_AA
    AA = self.activation(AA)
    PA = tf.matmul(pair_features, self.W_PA) + self.b_PA
    PA = self.activation(PA)
    PA = tf.segment_sum(PA, pair_split)

    A = tf.matmul(tf.concat([AA, PA], 1), self.W_A) + self.b_A
    A = self.activation(A)

    if self.update_pair:
      AP_ij = tf.matmul(
          tf.reshape(
              tf.gather(atom_features, atom_to_pair),
              [-1, 2 * self.n_atom_input_feat]), self.W_AP) + self.b_AP
      AP_ij = self.activation(AP_ij)
      AP_ji = tf.matmul(
          tf.reshape(
              tf.gather(atom_features, tf.reverse(atom_to_pair, [1])),
              [-1, 2 * self.n_atom_input_feat]), self.W_AP) + self.b_AP
      AP_ji = self.activation(AP_ji)

      PP = tf.matmul(pair_features, self.W_PP) + self.b_PP
      PP = self.activation(PP)
      P = tf.matmul(tf.concat([AP_ij + AP_ji, PP], 1), self.W_P) + self.b_P
      P = self.activation(P)
    else:
      P = pair_features

    return A, P

# TODO(rbharath): This class does not yet have a
# TensorGraph equivalent, but one may not be required.
# Commented out for now, remove if OK.
class WeaveConcat(Layer):
  """" Concat a batch of molecules into a batch of atoms
  """

  def __init__(self,
               batch_size,
               n_atom_input_feat=50,
               n_output=128,
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    """
    Parameters
    ----------
    batch_size: int
      number of molecules in a batch
    n_atom_input_feat: int, optional
      Number of features for each atom in input.
    n_output: int, optional
      Number of output features for each atom(concatenated)
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied

    """
    self.batch_size = batch_size
    self.n_atom_input_feat = n_atom_input_feat
    self.n_output = n_output
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    super(WeaveConcat, self).__init__(**kwargs)

  def build(self):
    """"Construct internal trainable weights.
    """

    self.W = self.init([self.n_atom_input_feat, self.n_output])
    self.b = model_ops.zeros(shape=[
        self.n_output,
    ])

    self.trainable_weights = self.W + self.b

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [atom_features, atom_mask]

    Parameters
    ----------
    x: list
      Tensors as listed above
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    outputs: Tensor
      Tensor of concatenated atom features
    """
    self.build()
    atom_features = x[0]
    atom_masks = x[1]
    A = tf.split(atom_features, self.batch_size, axis=0)
    A_mask = tf.split(
        tf.cast(atom_masks, dtype=tf.bool), self.batch_size, axis=0)
    outputs = tf.concat(
        [tf.boolean_mask(A[i], A_mask[i]) for i in range(len(A))], axis=0)
    outputs = tf.matmul(outputs, self.W) + self.b
    outputs = self.activation(outputs)
    return outputs

# TODO(rbharath): This class does not yet have a
# TensorGraph equivalent, but one may not be required.
# Commented out for now, remove if OK.
class AlternateWeaveGather(WeaveGather):
  """Alternate implementation of weave gather layer
     corresponding to AlternateWeaveLayer
  """

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [atom_features, atom_split]

    Parameters
    ----------
    x: list
      Tensors as listed above
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    outputs: Tensor
      Tensor of molecular features
    """
    # Add trainable weights
    self.build()
    outputs = x[0]
    atom_split = x[1]

    if self.gaussian_expand:
      outputs = self.gaussian_histogram(outputs)

    output_molecules = tf.segment_sum(outputs, atom_split)

    if self.gaussian_expand:
      output_molecules = tf.matmul(output_molecules, self.W) + self.b
      output_molecules = self.activation(output_molecules)
    return output_molecules
