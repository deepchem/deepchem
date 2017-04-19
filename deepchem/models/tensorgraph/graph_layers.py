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

from deepchem.models.tensorgraph.layers import Layer

class Combine_AP(Layer):
  def __init__(self, **kwargs):
    super(Combine_AP, self).__init__(**kwargs)
  
  def _create_tensor(self):
    A = self.in_layers[0].out_tensor
    P = self.in_layers[1].out_tensor
    self.out_tensor = [A, P]

class Separate_AP(Layer):
  def __init__(self, **kwargs):
    super(Separate_AP, self).__init__(**kwargs)
  
  def _create_tensor(self):
    self.out_tensor = self.in_layers[0].out_tensor[0]
    
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
               update_pair=True,
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
    super(WeaveLayer, self).__init__(**kwargs)

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

  def _create_tensor(self):
    self.build()

    atom_features = self.in_layers[0].out_tensor[0]
    pair_features = self.in_layers[0].out_tensor[1]

    pair_split = self.in_layers[1].out_tensor
    atom_to_pair = self.in_layers[2].out_tensor

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
    self.out_tensor = [A, P]

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

  def  _create_tensor(self):
    # Add trainable weights
    self.build()
    outputs = self.in_layers[0].out_tensor
    atom_split = self.in_layers[1].out_tensor

    if self.gaussian_expand:
      outputs = self.gaussian_histogram(outputs)

    output_molecules = tf.segment_sum(outputs, atom_split)
    
    if self.gaussian_expand:
      output_molecules = tf.matmul(output_molecules, self.W) + self.b
      output_molecules = self.activation(output_molecules)
    self.out_tensor = output_molecules


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


class DTNNEmbedding(Layer):
  """Generate embeddings for all atoms in the batch
  """

  def __init__(self,
               n_embedding=30,
               periodic_table_length=83,
               init='glorot_uniform',
               **kwargs):
    self.n_embedding = n_embedding
    self.periodic_table_length = periodic_table_length
    self.init = initializations.get(init)  # Set weight initialization

    super(DTNNEmbedding, self).__init__(**kwargs)

  def build(self):

    self.embedding_list = self.init(
        [self.periodic_table_length, self.n_embedding])
    self.trainable_weights = [self.embedding_list]

  def _create_tensor(self):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: Tensor 
      1D tensor of length n_atoms (atomic number)

    Returns
    -------
    tf.Tensor
      Of shape (n_atoms, n_embedding), where n_embedding is number of atom features
    """
    self.build()
    atom_number = self.in_layers[0].out_tensor
    atom_features = tf.nn.embedding_lookup(self.embedding_list, atom_number)
    self.out_tensor = atom_features


class DTNNStep(Layer):
  """A convolution step that merge in distance and atom info of 
     all other atoms into current atom.
   
     model based on https://arxiv.org/abs/1609.08259
  """

  def __init__(self,
               n_embedding=30,
               n_distance=100,
               n_hidden=60,
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    self.n_embedding = n_embedding
    self.n_distance = n_distance
    self.n_hidden = n_hidden
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations

    super(DTNNStep, self).__init__(**kwargs)

  def build(self):
    self.W_cf = self.init([self.n_embedding, self.n_hidden])
    self.W_df = self.init([self.n_distance, self.n_hidden])
    self.W_fc = self.init([self.n_hidden, self.n_embedding])
    self.b_cf = model_ops.zeros(shape=[
        self.n_hidden,
    ])
    self.b_df = model_ops.zeros(shape=[
        self.n_hidden,
    ])
    #self.b_fc = model_ops.zeros(shape=[self.n_embedding,])

    self.trainable_weights = [
        self.W_cf, self.W_df, self.W_fc, self.b_cf, self.b_df
    ]

  def _create_tensor(self):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: list of Tensor 
      should be [atom_features: n_atoms*n_embedding, 
                 distance_matrix: n_pairs*n_distance,
                 atom_membership: n_atoms
                 distance_membership_i: n_pairs,
                 distance_membership_j: n_pairs,
                 ]

    Returns
    -------
    tf.Tensor
      new embeddings for atoms, same shape as x[0]
    """
    self.build()
    atom_features = self.in_layers[0].out_tensor
    distance = self.in_layers[1].out_tensor
    distance_membership_i = self.in_layers[2].out_tensor
    distance_membership_j = self.in_layers[3].out_tensor
    distance_hidden = tf.matmul(distance, self.W_df) + self.b_df
    #distance_hidden = self.activation(distance_hidden)
    atom_features_hidden = tf.matmul(atom_features, self.W_cf) + self.b_cf
    #atom_features_hidden = self.activation(atom_features_hidden)
    outputs = tf.multiply(distance_hidden,
                          tf.gather(atom_features_hidden,
                                    distance_membership_j))

    # for atom i in a molecule m, this step multiplies together distance info of atom pair(i,j)
    # and embeddings of atom j(both gone through a hidden layer)
    outputs = tf.matmul(outputs, self.W_fc)
    outputs = self.activation(outputs)

    output_ii = tf.multiply(self.b_df, atom_features_hidden)
    output_ii = tf.matmul(output_ii, self.W_fc)
    output_ii = self.activation(output_ii)

    # for atom i, sum the influence from all other atom j in the molecule
    outputs = tf.segment_sum(outputs,
                             distance_membership_i) - output_ii + atom_features
    self.out_tensor = outputs


class DTNNGather(Layer):
  """Map the atomic features into molecular properties and sum
  """

  def __init__(self,
               n_embedding=30,
               n_outputs=100,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    self.n_embedding = n_embedding
    self.n_outputs = n_outputs
    self.layer_sizes = layer_sizes
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations

    super(DTNNGather, self).__init__(**kwargs)

  def build(self):
    self.W_list = []
    self.b_list = []
    prev_layer_size = self.n_embedding
    for i, layer_size in enumerate(self.layer_sizes):
      self.W_list.append(self.init([prev_layer_size, layer_size]))
      self.b_list.append(model_ops.zeros(shape=[
          layer_size,
      ]))
      prev_layer_size = layer_size
    self.W_list.append(self.init([prev_layer_size, self.n_outputs]))
    self.b_list.append(model_ops.zeros(shape=[
        self.n_outputs,
    ]))
    prev_layer_size = self.n_outputs

    self.trainable_weights = self.W_list + self.b_list

  def _create_tensor(self):
    """Execute this layer on input tensors.

    Parameters
    ----------
    x: list of Tensor 
      should be [embedding tensor of molecules, of shape (batch_size*max_n_atoms*n_embedding),
                 mask tensor of molecules, of shape (batch_size*max_n_atoms)]

    Returns
    -------
    list of tf.Tensor
      Of shape (batch_size)
    """
    self.build()
    output = self.in_layers[0].out_tensor
    atom_membership = self.in_layers[1].out_tensor
    for i, W in enumerate(self.W_list):
      output = tf.matmul(output, W) + self.b_list[i]
      output = self.activation(output)
    output = tf.segment_sum(output, atom_membership)
    self.out_tensor = output