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
from deepchem.models.tensorgraph.layers import convert_to_layers


class Combine_AP(Layer):

  def __init__(self, **kwargs):
    super(Combine_AP, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    A = in_layers[0].out_tensor
    P = in_layers[1].out_tensor
    out_tensor = [A, P]
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class Separate_AP(Layer):

  def __init__(self, **kwargs):
    super(Separate_AP, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    out_tensor = in_layers[0].out_tensor[0]
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class WeaveLayer(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.WeaveLayer

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
    self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P = None, None, None, None, None, None
    super(WeaveLayer, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.

        TODO(rbharath): Need to make this not set instance variables to
        follow style in other layers.
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

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ description and explanation refer to deepchem.nn.WeaveLayer
        parent layers: [atom_features, pair_features], pair_split, atom_to_pair
        """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()

    atom_features = in_layers[0].out_tensor[0]
    pair_features = in_layers[0].out_tensor[1]

    pair_split = in_layers[1].out_tensor
    atom_to_pair = in_layers[2].out_tensor

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
    out_tensor = [A, P]
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

  def none_tensors(self):
    W_AP, b_AP, W_PP, W_PP, W_P, b_P = self.W_AP, self.b_AP, self.W_PP, self.W_PP, self.W_P, self.b_P
    self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P = None, None, None, None, None, None

    W_AA, b_AA, W_PA, b_PA, W_A, b_A = self.W_AA, self.b_AA, self.W_PA, self.b_PA, self.W_A, self.b_A
    self.W_AA, self.b_AA, self.W_PA, self.b_PA, self.W_A, self.b_A = None, None, None, None, None, None

    out_tensor, trainable_weights, variables = self.out_tensor, self.trainable_weights, self.variables
    self.out_tensor, self.trainable_weights, self.variables, self.activation, self.init = None, [], [], None, None

    return W_AP, b_AP, W_PP, W_PP, W_P, b_P, \
           W_AA, b_AA, W_PA, b_PA, W_A, b_A, \
           out_tensor, trainable_weights, variables

  def set_tensors(self, tensor):
    self.W_AP, self.b_AP, self.W_PP, self.W_PP, self.W_P, self.b_P, \
    self.W_AA, self.b_AA, self.W_PA, self.b_PA, self.W_A, self.b_A, \
    self.out_tensor, self.trainable_weights, self.variables = tensor


class WeaveGather(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.WeaveGather
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
    self.W, self.b = None, None
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

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ description and explanation refer to deepchem.nn.WeaveGather
        parent layers: atom_features, atom_split
        """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    outputs = in_layers[0].out_tensor
    atom_split = in_layers[1].out_tensor

    if self.gaussian_expand:
      outputs = self.gaussian_histogram(outputs)

    output_molecules = tf.segment_sum(outputs, atom_split)

    if self.gaussian_expand:
      output_molecules = tf.matmul(output_molecules, self.W) + self.b
      output_molecules = self.activation(output_molecules)
    out_tensor = output_molecules
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

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

  def none_tensors(self):
    W, b = self.W, self.b
    self.W, self.b = None, None

    out_tensor, trainable_weights, variables = self.out_tensor, self.trainable_weights, self.variables
    self.out_tensor, self.trainable_weights, self.variables = None, [], []
    return W, b, out_tensor, trainable_weights, variables

  def set_tensors(self, tensor):
    self.W, self.b, self.out_tensor, self.trainable_weights, self.variables = tensor


class DTNNEmbedding(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.DTNNEmbedding
    """

  def __init__(self,
               n_embedding=30,
               periodic_table_length=83,
               init='glorot_uniform',
               **kwargs):
    """
        Parameters
        ----------
        n_embedding: int, optional
          Number of features for each atom
        periodic_table_length: int, optional
          Length of embedding, 83=Bi
        init: str, optional
          Weight initialization for filters.
        """
    self.n_embedding = n_embedding
    self.periodic_table_length = periodic_table_length
    self.init = initializations.get(init)  # Set weight initialization

    super(DTNNEmbedding, self).__init__(**kwargs)

  def build(self):
    self.embedding_list = self.init(
        [self.periodic_table_length, self.n_embedding])
    self.trainable_weights = [self.embedding_list]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """description and explanation refer to deepchem.nn.DTNNEmbedding
        parent layers: atom_number
        """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    atom_number = in_layers[0].out_tensor
    atom_features = tf.nn.embedding_lookup(self.embedding_list, atom_number)
    self.out_tensor = atom_features


class DTNNStep(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.DTNNStep
    """

  def __init__(self,
               n_embedding=30,
               n_distance=100,
               n_hidden=60,
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    """
        Parameters
        ----------
        n_embedding: int, optional
          Number of features for each atom
        n_distance: int, optional
          granularity of distance matrix
        n_hidden: int, optional
          Number of nodes in hidden layer
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        """
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

    self.trainable_weights = [
        self.W_cf, self.W_df, self.W_fc, self.b_cf, self.b_df
    ]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """description and explanation refer to deepchem.nn.DTNNStep
        parent layers: atom_features, distance, distance_membership_i, distance_membership_j
        """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    atom_features = in_layers[0].out_tensor
    distance = in_layers[1].out_tensor
    distance_membership_i = in_layers[2].out_tensor
    distance_membership_j = in_layers[3].out_tensor
    distance_hidden = tf.matmul(distance, self.W_df) + self.b_df
    atom_features_hidden = tf.matmul(atom_features, self.W_cf) + self.b_cf
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
    out_tensor = outputs
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor


class DTNNGather(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.DTNNGather
    """

  def __init__(self,
               n_embedding=30,
               n_outputs=100,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='tanh',
               **kwargs):
    """
        Parameters
        ----------
        n_embedding: int, optional
          Number of features for each atom
        n_outputs: int, optional
          Number of features for each molecule(output)
        layer_sizes: list of int, optional(default=[1000])
          Structure of hidden layer(s)
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        """
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

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """description and explanation refer to deepchem.nn.DTNNGather
        parent layers: atom_features, atom_membership
        """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    output = in_layers[0].out_tensor
    atom_membership = in_layers[1].out_tensor
    for i, W in enumerate(self.W_list[:-1]):
      output = tf.matmul(output, W) + self.b_list[i]
      output = self.activation(output)
    output = tf.matmul(output, self.W_list[-1]) + self.b_list[-1]
    output = tf.segment_sum(output, atom_membership)
    out_tensor = output
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor


class DTNNExtract(Layer):

  def __init__(self, task_id, **kwargs):
    self.task_id = task_id
    super(DTNNExtract, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    output = in_layers[0].out_tensor
    out_tensor = output[:, self.task_id:self.task_id + 1]
    self.out_tensor = out_tensor
    return out_tensor


class DAGLayer(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.DAGLayer
    """

  def __init__(self,
               n_graph_feat=30,
               n_atom_feat=75,
               max_atoms=50,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               batch_size=64,
               **kwargs):
    """
        Parameters
        ----------
        n_graph_feat: int, optional
          Number of features for each node(and the whole grah).
        n_atom_feat: int, optional
          Number of features listed per atom.
        max_atoms: int, optional
          Maximum number of atoms in molecules.
        layer_sizes: list of int, optional(default=[1000])
          Structure of hidden layer(s)
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        dropout: float, optional
          Dropout probability, not supported here
        batch_size: int, optional
          number of molecules in a batch
        """
    super(DAGLayer, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.batch_size = batch_size
    self.n_inputs = n_atom_feat + (self.max_atoms - 1) * n_graph_feat
    # number of inputs each step
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_graph_feat
    self.n_atom_feat = n_atom_feat

  def build(self):
    """"Construct internal trainable weights.
        """

    self.W_list = []
    self.b_list = []
    prev_layer_size = self.n_inputs
    for layer_size in self.layer_sizes:
      self.W_list.append(self.init([prev_layer_size, layer_size]))
      self.b_list.append(model_ops.zeros(shape=[
          layer_size,
      ]))
      prev_layer_size = layer_size
    self.W_list.append(self.init([prev_layer_size, self.n_outputs]))
    self.b_list.append(model_ops.zeros(shape=[
        self.n_outputs,
    ]))

    self.trainable_weights = self.W_list + self.b_list

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """description and explanation refer to deepchem.nn.DAGLayer
        parent layers: atom_features, parents, calculation_orders, calculation_masks, n_atoms
        """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    # Add trainable weights
    self.build()

    atom_features = in_layers[0].out_tensor
    # each atom corresponds to a graph, which is represented by the `max_atoms*max_atoms` int32 matrix of index
    # each gragh include `max_atoms` of steps(corresponding to rows) of calculating graph features
    parents = in_layers[1].out_tensor
    # target atoms for each step: (batch_size*max_atoms) * max_atoms
    calculation_orders = in_layers[2].out_tensor
    calculation_masks = in_layers[3].out_tensor

    n_atoms = in_layers[4].out_tensor
    # initialize graph features for each graph
    graph_features_initial = tf.zeros((self.max_atoms * self.batch_size,
                                       self.max_atoms + 1, self.n_graph_feat))
    # initialize graph features for each graph
    # another row of zeros is generated for padded dummy atoms
    graph_features = tf.Variable(graph_features_initial, trainable=False)

    for count in range(self.max_atoms):
      # `count`-th step
      # extracting atom features of target atoms: (batch_size*max_atoms) * n_atom_features
      mask = calculation_masks[:, count]
      current_round = tf.boolean_mask(calculation_orders[:, count], mask)
      batch_atom_features = tf.gather(atom_features, current_round)

      # generating index for graph features used in the inputs
      index = tf.stack(
          [
              tf.reshape(
                  tf.stack(
                      [tf.boolean_mask(tf.range(n_atoms), mask)] *
                      (self.max_atoms - 1),
                      axis=1), [-1]),
              tf.reshape(tf.boolean_mask(parents[:, count, 1:], mask), [-1])
          ],
          axis=1)
      # extracting graph features for parents of the target atoms, then flatten
      # shape: (batch_size*max_atoms) * [(max_atoms-1)*n_graph_features]
      batch_graph_features = tf.reshape(
          tf.gather_nd(graph_features, index),
          [-1, (self.max_atoms - 1) * self.n_graph_feat])

      # concat into the input tensor: (batch_size*max_atoms) * n_inputs
      batch_inputs = tf.concat(
          axis=1, values=[batch_atom_features, batch_graph_features])
      # DAGgraph_step maps from batch_inputs to a batch of graph_features
      # of shape: (batch_size*max_atoms) * n_graph_features
      # representing the graph features of target atoms in each graph
      batch_outputs = self.DAGgraph_step(batch_inputs, self.W_list, self.b_list)

      # index for targe atoms
      target_index = tf.stack([tf.range(n_atoms), parents[:, count, 0]], axis=1)
      target_index = tf.boolean_mask(target_index, mask)
      # update the graph features for target atoms
      graph_features = tf.scatter_nd_update(graph_features, target_index,
                                            batch_outputs)

    out_tensor = batch_outputs
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

  def DAGgraph_step(self, batch_inputs, W_list, b_list):
    outputs = batch_inputs
    for idw, W in enumerate(W_list):
      outputs = tf.nn.xw_plus_b(outputs, W, b_list[idw])
      outputs = self.activation(outputs)
    return outputs


class DAGGather(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.DAGGather
    """

  def __init__(self,
               n_graph_feat=30,
               n_outputs=30,
               max_atoms=50,
               layer_sizes=[100],
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               **kwargs):
    """
        Parameters
        ----------
        n_graph_feat: int, optional
          Number of features for each atom
        n_outputs: int, optional
          Number of features for each molecule.
        max_atoms: int, optional
          Maximum number of atoms in molecules.
        layer_sizes: list of int, optional
          Structure of hidden layer(s)
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied
        dropout: float, optional
          Dropout probability, not supported
        """
    super(DAGGather, self).__init__(**kwargs)

    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs

  def build(self):
    """"Construct internal trainable weights.
        """

    self.W_list = []
    self.b_list = []
    prev_layer_size = self.n_graph_feat
    for layer_size in self.layer_sizes:
      self.W_list.append(self.init([prev_layer_size, layer_size]))
      self.b_list.append(model_ops.zeros(shape=[
          layer_size,
      ]))
      prev_layer_size = layer_size
    self.W_list.append(self.init([prev_layer_size, self.n_outputs]))
    self.b_list.append(model_ops.zeros(shape=[
        self.n_outputs,
    ]))

    self.trainable_weights = self.W_list + self.b_list

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """description and explanation refer to deepchem.nn.DAGGather
        parent layers: atom_features, membership
        """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    # Add trainable weights
    self.build()

    # Extract atom_features
    atom_features = in_layers[0].out_tensor
    membership = in_layers[1].out_tensor
    # Extract atom_features
    graph_features = tf.segment_sum(atom_features, membership)
    # sum all graph outputs
    outputs = self.DAGgraph_step(graph_features, self.W_list, self.b_list)
    out_tensor = outputs
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

  def DAGgraph_step(self, batch_inputs, W_list, b_list):
    outputs = batch_inputs
    for idw, W in enumerate(W_list):
      outputs = tf.nn.xw_plus_b(outputs, W, b_list[idw])
      outputs = self.activation(outputs)
    return outputs
