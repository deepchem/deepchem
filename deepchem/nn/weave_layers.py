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

  Detailed structure and explanations:
  https://arxiv.org/abs/1603.00856

  """

  def __init__(self,
               max_atoms,
               n_atom_input_feat=75,
               n_pair_input_feat=14,
               n_atom_output_feat=50,
               n_pair_output_feat=50,
               n_hidden_AA=50,
               n_hidden_PA=50,
               n_hidden_AP=50,
               n_hidden_PP=50,
               update_pair=True,
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               **kwargs):
    """
    Parameters
    ----------
    max_atoms: int
      Maximum number of atoms in a molecule, should be defined based on dataset
    n_atom_input_feat: int, optional
      Number of features for each atom in input.
    n_pair_input_feat: int, optional
      Number of features for each pair of atoms in input.
    n_atom_output_feat: int, optional
      Number of features for each atom in output.
    n_pair_output_feat: int, optional
      Number of features for each pair of atoms in output.
    n_hidden_XX: int, optional
      Number of units(convolution depths) in corresponding hidden layer
    update_pair: bool, optional
      Whether to calculate for pair features,
      could be turned off for last layer
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    dropout: float, optional
      Dropout probability, not supported here

    """
    super(WeaveLayer, self).__init__(**kwargs)
    self.max_atoms = max_atoms
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.update_pair = update_pair  # last weave layer does not need to update
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

    self.trainable_weights = [
        self.W_AA, self.b_AA, self.W_PA, self.b_PA, self.W_A, self.b_A
    ]
    if self.update_pair:
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

      self.trainable_weights.extend(
          [self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P])

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
    max_atoms = self.max_atoms

    AA = tf.tensordot(atom_features, self.W_AA, [[2], [0]]) + self.b_AA
    AA = self.activation(AA)
    PA = tf.reduce_sum(
        tf.tensordot(pair_features, self.W_PA, [[3], [0]]) + self.b_PA, axis=2)
    PA = self.activation(PA)
    A = tf.tensordot(tf.concat([AA, PA], 2), self.W_A, [[2], [0]]) + self.b_A
    A = self.activation(A)
    A = tf.multiply(A, tf.expand_dims(atom_mask, axis=2))

    if self.update_pair:
      AP_combine = tf.concat([
          tf.stack([atom_features] * max_atoms, axis=2),
          tf.stack([atom_features] * max_atoms, axis=1)
      ], 3)
      AP_combine_t = tf.transpose(AP_combine, perm=[0, 2, 1, 3])
      AP = tf.tensordot(AP_combine + AP_combine_t, self.W_AP, [[3], [0]
                                                              ]) + self.b_AP
      AP = self.activation(AP)
      PP = tf.tensordot(pair_features, self.W_PP, [[3], [0]]) + self.b_PP
      PP = self.activation(PP)
      P = tf.tensordot(tf.concat([AP, PP], 3), self.W_P, [[3], [0]]) + self.b_P
      P = self.activation(P)
      P = tf.multiply(P, tf.expand_dims(pair_mask, axis=3))
    else:
      P = pair_features

    return A, P


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


class WeaveGather(Layer):
  """" Gather layer of Weave model
  a batch of normalized atom features go through a hidden layer,
  then summed to form molecular features
  """

  def __init__(self,
               batch_size,
               n_input=128,
               gaussian_expand=False,
               init='glorot_uniform',
               activation='tanh',
               epsilon=1e-3,
               momentum=0.99,
               **kwargs):
    """
    Parameters
    ----------
    batch_size: int
      number of molecules in a batch
    n_input: int, optional
      number of features for each input molecule
    gaussian_expand: boolean. optional
      Whether to expand each dimension of atomic features by gaussian histogram
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied

    """
    self.n_input = n_input
    self.batch_size = batch_size
    self.gaussian_expand = gaussian_expand
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.epsilon = epsilon
    self.momentum = momentum
    super(WeaveGather, self).__init__(**kwargs)

  def build(self):
    if self.gaussian_expand:
      self.W = self.init([self.n_input * 11, self.n_input])
      self.b = model_ops.zeros(shape=[
          self.n_input,
      ])
      self.trainable_weights = self.W + self.b
    else:
      self.trainable_weights = None

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [atom_features, membership]

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
    membership = x[1]

    if self.gaussian_expand:
      outputs = self.gaussian_histogram(outputs)

    outputs = tf.dynamic_partition(outputs, membership, self.batch_size)

    output_molecules = [tf.reduce_sum(molecule, 0) for molecule in outputs]

    output_molecules = tf.stack(output_molecules)
    if self.gaussian_expand:
      output_molecules = tf.matmul(output_molecules, self.W) + self.b
      output_molecules = self.activation(output_molecules)
    return output_molecules

  def gaussian_histogram(self, x):
    gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                            (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                            (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                            (1.080, 0.170), (1.645, 0.283)]
    dist = [
        tf.contrib.distributions.Normal(p[0], p[1])
        for p in gaussian_memberships
    ]
    dist_max = [dist[i].prob(gaussian_memberships[i][0]) for i in range(11)]
    outputs = [dist[i].prob(x) / dist_max[i] for i in range(11)]
    outputs = tf.stack(outputs, axis=2)
    outputs = outputs / tf.reduce_sum(outputs, axis=2, keep_dims=True)
    outputs = tf.reshape(outputs, [-1, self.n_input * 11])
    return outputs


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
