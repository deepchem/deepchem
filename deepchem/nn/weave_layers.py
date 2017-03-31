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
from deepchem.nn.copy import Layer


class WeaveLayer(Layer):
  """" Main layer of Weave model
  For each molecule, atom features and pair features are recombined to 
  generate new atom(pair) features
  """

  def __init__(self,
               n_atom_input_feat=75,
               n_pair_input_feat=14,
               n_atom_output_feat=50,
               n_pair_output_feat=50,
               n_hidden_AA=50,
               n_hidden_PA=50,
               n_hidden_AP=50,
               n_hidden_PP=50,
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               **kwargs):
    """
    Parameters
    ----------
    n_atom_input_feat: int
      Number of features for each atom in input.
    n_pair_input_feat: int
      Number of features for each pair of atoms in input.
    n_atom_output_feat: int
      Number of features for each atom in output.
    n_pair_output_feat: int
      Number of features for each pair of atoms in output.
    n_hidden_XX: int
      Number of units(convolution depths) in corresponding hidden layer
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    dropout: float, optional
      Dropout probability, not supported here

    """
    super(WeaveLayer, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.n_hidden_AA = n_hidden_AA
    self.n_hidden_PA = n_hidden_PA
    self.n_hidden_AP = n_hidden_AP
    self.n_hidden_PP = n_hidden_PP
    self.n_hidden_A = n_hidden_AA + n_hidden_PA
    self.n_hidden_P = n_hidden_AP + n_hidden_PP

    self.n_atom_input_feat = n_atom_input_feat
    self.n_pair_input_feat = n_pair_input_feat
    self.n_atom_output_feat = n_atom_output_feat
    self.n_pair_output_feat = n_pair_output_feat

  def build(self):
    """"Construct internal trainable weights.
    """

    self.W_AA = self.init([self.n_atom_input_feat, self.n_hidden_AA])
    self.b_AA = model_ops.zeros(shape=[
        self.n_hidden_AA,
    ])

    self.W_PA = self.init([self.n_pair_input_feat, self.n_hidden_PA])
    self.b_PA = model_ops.zeros(shape=[
        self.n_hidden_PA,
    ])

    self.W_A = self.init([self.n_hidden_A, self.n_atom_output_feat])
    self.b_A = model_ops.zeros(shape=[
        self.n_atom_output_feat,
    ])

    self.W_AP = self.init([self.n_atom_input_feat * 2, self.n_hidden_AP])
    self.b_AP = model_ops.zeros(shape=[
        self.n_hidden_AP,
    ])

    self.W_PP = self.init([self.n_pair_input_feat, self.n_hidden_PP])
    self.b_PP = model_ops.zeros(shape=[
        self.n_hidden_PP,
    ])

    self.W_P = self.init([self.n_hidden_P, self.n_pair_output_feat])
    self.b_P = model_ops.zeros(shape=[
        self.n_pair_output_feat,
    ])

    self.trainable_weights = self.W_AA + self.b_AA + self.W_PA + self.b_PA + \
        self.W_A + self.b_A + self.W_AP + self.b_AP + self.W_PP + self.b_PP + \
        self.W_P + self.b_P

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [atom_features, pair_features, atom_mask, pair_mask]
    
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

    atom_mask = x[2]
    pair_mask = x[3]
    max_atoms = atom_features.get_shape().as_list()[1]

    AA = tf.tensordot(atom_features, self.W_AA, [[2], [0]]) + self.b_AA
    PA = tf.reduce_sum(
        tf.tensordot(pair_features, self.W_PA, [[3], [0]]) + self.b_PA, axis=2)
    A = tf.tensordot(tf.concat([AA, PA], 2), self.W_A, [[2], [0]]) + self.b_A
    AP_combine = tf.concat([
        tf.stack([atom_features] * max_atoms, axis=2),
        tf.stack([atom_features] * max_atoms, axis=1)
    ], 3)
    AP_combine_t = tf.transpose(AP_combine, perm=[0, 2, 1, 3])
    AP = tf.tensordot(AP_combine + AP_combine_t, self.W_AP,
                      [[3], [0]]) + self.b_AP
    PP = tf.tensordot(pair_features, self.W_PP, [[3], [0]]) + self.b_PP
    P = tf.tensordot(tf.concat([AP, PP], 3), self.W_P, [[3], [0]]) + self.b_P

    A = tf.multiply(A, tf.expand_dims(atom_mask, axis=2))
    P = tf.multiply(P, tf.expand_dims(pair_mask, axis=3))
    return A, P

class WeaveConcat(Layer):
  """" Concat a batch of molecules into a batch of atoms
  """

  def __init__(self,
               **kwargs):
    super(WeaveConcat, self).__init__(**kwargs)

  def build(self, shape):
    pass

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: list
      Tensors: atom_features, atom_masks
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
    A = tf.unstack(atom_features, axis=0)
    A_mask = tf.unstack(tf.cast(atom_masks, dtype=tf.bool), axis=0)
    outputs = tf.concat([tf.boolean_mask(A[i], A_mask[i]) for i in range(len(A))], axis=0)
    return outputs

class WeaveGather(Layer):
  """" Gather layer of Weave model
  a batch of normalized atom features go through a hidden layer, 
  then summed to form molecular features
  """

  def __init__(self,
               n_atom_input_feat=50,
               n_hidden=128,
               init='glorot_uniform',
               activation='tanh',
               gaussian_expand=True,
               dropout=None,
               epsilon=1e-3,
               momentum=0.99,
               **kwargs):
    """
    Parameters
    ----------
    n_atom_input_feat: int
      Number of features for each atom in input.
    n_pair_input_feat: int
      Number of features for each pair of atoms in input.
    n_atom_output_feat: int
      Number of features for each atom in output.
    n_pair_output_feat: int
      Number of features for each pair of atoms in output.
    n_hidden_XX: int
      Number of units(convolution depths) in corresponding hidden layer
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    dropout: float, optional
      Dropout probability, not supported here
    gaussian_expand: boolean. optional
      Whether to expand each dimension of atomic features by gaussian histogram

    """
    super(WeaveLayer, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.n_hidden = n_hidden
    self.n_atom_input_feat = n_atom_input_feat
    self.gaussian_expand = gaussian_expand
    if gaussian_expand:
      self.n_outputs = self.n_hidden * 11
    else:
      self.n_outputs = self.n_hidden
    self.epsilon = epsilon
    self.momentum = momentum

  def build(self, shape):
    """"Construct internal trainable weights.
    """

    self.W = self.init([self.n_atom_input_feat, self.n_hidden])
    self.b = model_ops.zeros(shape=[self.n_hidden,])

    self.trainable_weights = self.W + self.b

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: list
      Tensors: atom_features, atom_masks
    mask: bool, optional
      Ignored. Present only to shadow superclass call() method.

    Returns
    -------
    outputs: Tensor
      Tensor of molecular features
    """
    # Add trainable weights
    self.build()
    atom_features = x[0]
    atom_masks = x[1]
    outputs = tf.matmul(atom_features, self.W) + self.b
    if self.gaussian_expand:
      outputs = self.gaussian_histogram(outputs)
      
    outputs = tf.reduce_sum(outputs, axis=1)
    return outputs

  def gaussian_histogram(x):
      
    return x
