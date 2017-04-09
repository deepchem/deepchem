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

    self.trainable_weights = [
        self.W_AA, self.b_AA, self.W_PA, self.b_PA, self.W_A, self.b_A,
        self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P
    ]

  def call(self, x, mask=None):
    """Execute this layer on input tensors.

    x = [atom_features, pair_features, pair_split, pair_membership, atom_split]
    
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
    pair_membership = x[3]
    atom_split = x[4]
    atom_to_pair = x[5]

    AA = tf.matmul(atom_features, self.W_AA) + self.b_AA
    AA = self.activation(AA)

    PA = tf.matmul(pair_features, self.W_PA) + self.b_PA
    PA = self.activation(PA)
    PAs = tf.split(PA, pair_split, axis=0)
    PA = [tf.reduce_sum(molecule, 0) for molecule in PAs]
    PA = tf.boolean_mask(PA, pair_membership)
    
    A = tf.matmul(tf.concat([AA, PA], 1), self.W_A) + self.b_A
    A = self.activation(A)

    AP_ij = tf.matmul(tf.reshape(tf.gather(atom_features, atom_to_pair), 
                                 [-1, 2*self.n_atom_input_feat]), self.W_AP) + self.b_AP
    AP_ij = self.activation(AP_ij)
    AP_ji = tf.matmul(tf.reshape(tf.gather(atom_features, tf.reverse(atom_to_pair, [1])), 
                                 [-1, 2*self.n_atom_input_feat]), self.W_AP) + self.b_AP
    AP_ji = self.activation(AP_ji)
    
    PP = tf.matmul(pair_features, self.W_PP) + self.b_PP
    PP = self.activation(PP)
    
    P = tf.matmul(tf.concat([AP_ij + AP_ji, PP], 1), self.W_P) + self.b_P
    P = self.activation(P)
    
    return A, P


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
    gaussian_expand: boolean. optional
      Whether to expand each dimension of atomic features by gaussian histogram

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

    outputs = tf.split(outputs, atom_split, axis=0)

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
        tf.contrib.distributions.Normal(mu=p[0], sigma=p[1])
        for p in gaussian_memberships
    ]
    dist_max = [dist[i].pdf(gaussian_memberships[i][0]) for i in range(11)]
    outputs = [dist[i].pdf(x) / dist_max[i] for i in range(11)]
    outputs = tf.stack(outputs, axis=2)
    outputs = outputs / tf.reduce_sum(outputs, axis=2, keep_dims=True)
    outputs = tf.reshape(outputs, [-1, self.n_input * 11])
    return outputs
