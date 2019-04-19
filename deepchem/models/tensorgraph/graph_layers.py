#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:02:04 2017

@author: michael
"""
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem.models.layers
from deepchem.models.tensorgraph import activations
from deepchem.models.tensorgraph import initializations
from deepchem.models.tensorgraph import model_ops
from deepchem.models.tensorgraph.layers import Layer, LayerSplitter, KerasLayer
from deepchem.models.tensorgraph.layers import convert_to_layers


class WeaveLayer(KerasLayer):
  """ TensorGraph style implementation
  Note: Use WeaveLayerFactory to construct this layer
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
               **kwargs):
    """
    Parameters
    ----------
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
    """
    super(WeaveLayer, self).__init__(**kwargs)
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.update_pair = update_pair  # last weave layer does not need to update
    self.n_hidden_AA = n_hidden_AA
    self.n_hidden_PA = n_hidden_PA
    self.n_hidden_AP = n_hidden_AP
    self.n_hidden_PP = n_hidden_PP

    self.n_atom_input_feat = n_atom_input_feat
    self.n_pair_input_feat = n_pair_input_feat
    self.n_atom_output_feat = n_atom_output_feat
    self.n_pair_output_feat = n_pair_output_feat
    self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P = None, None, None, None, None, None

  def _build_layer(self):
    return deepchem.models.layers.WeaveLayer(
        self.n_atom_input_feat, self.n_pair_input_feat, self.n_atom_output_feat,
        self.n_pair_output_feat, self.n_hidden_AA, self.n_hidden_PA,
        self.n_hidden_AP, self.n_hidden_PP, self.update_pair, self.init,
        self.activation)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """Creates weave tensors.

    parent layers: [atom_features, pair_features], pair_split, atom_to_pair
    """
    output = super(WeaveLayer, self).create_tensor(
        in_layers=in_layers, set_tensors=set_tensors, **kwargs)
    if set_tensors:
      self.out_tensors = output
      self.out_tensor = output[0]
      self._non_pickle_fields.append('out_tensors')
    return output


def WeaveLayerFactory(**kwargs):
  weaveLayer = WeaveLayer(**kwargs)
  return [LayerSplitter(i, in_layers=weaveLayer) for i in range(2)]


class WeaveGather(KerasLayer):
  """ TensorGraph style implementation
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
    self.init = init
    self.activation = activation
    self.epsilon = epsilon
    self.momentum = momentum
    self.W, self.b = None, None
    super(WeaveGather, self).__init__(**kwargs)

  def _build_layer(self):
    return deepchem.models.layers.WeaveGather(
        self.batch_size, self.n_input, self.gaussian_expand, self.init,
        self.activation, self.epsilon, self.momentum)


class DTNNEmbedding(KerasLayer):
  """ TensorGraph style implementation
  """

  def __init__(self,
               n_embedding=30,
               periodic_table_length=30,
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
    self.init = init
    super(DTNNEmbedding, self).__init__(**kwargs)

  def _build_layer(self):
    return deepchem.models.layers.DTNNEmbedding(
        self.n_embedding, self.periodic_table_length, self.init)


class DTNNStep(KerasLayer):
  """ TensorGraph style implementation
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
    self.init = init
    self.activation = activation
    super(DTNNStep, self).__init__(**kwargs)

  def _build_layer(self):
    return deepchem.models.layers.DTNNStep(self.n_embedding, self.n_distance,
                                           self.n_hidden, self.init,
                                           self.activation)


class DTNNGather(KerasLayer):
  """ TensorGraph style implementation
  """

  def __init__(self,
               n_embedding=30,
               n_outputs=100,
               layer_sizes=[100],
               output_activation=True,
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
    self.output_activation = output_activation
    self.init = init
    self.activation = activation
    super(DTNNGather, self).__init__(**kwargs)

  def _build_layer(self):
    return deepchem.models.layers.DTNNGather(
        self.n_embedding, self.n_outputs, self.layer_sizes,
        self.output_activation, self.init, self.activation)


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


class DAGLayer(KerasLayer):
  """ TensorGraph style implementation
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
        layer_sizes: list of int, optional(default=[100])
          List of hidden layer size(s):
          length of this list represents the number of hidden layers,
          and each element is the width of corresponding hidden layer.
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied.
        dropout: float, optional
          Dropout probability in hidden layer(s).
        batch_size: int, optional
          number of molecules in a batch.
        """
    super(DAGLayer, self).__init__(**kwargs)

    self.init = init
    self.activation = activation
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.batch_size = batch_size
    self.n_inputs = n_atom_feat + (self.max_atoms - 1) * n_graph_feat
    # number of inputs each step
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_graph_feat
    self.n_atom_feat = n_atom_feat

  def _build_layer(self):
    return deepchem.models.layers.DAGLayer(
        self.n_graph_feat, self.n_atom_feat, self.max_atoms, self.layer_sizes,
        self.init, self.activation, self.dropout, self.batch_size)


class DAGGather(KerasLayer):
  """ TensorGraph style implementation
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
          Number of features for each atom.
        n_outputs: int, optional
          Number of features for each molecule.
        max_atoms: int, optional
          Maximum number of atoms in molecules.
        layer_sizes: list of int, optional
          List of hidden layer size(s):
          length of this list represents the number of hidden layers,
          and each element is the width of corresponding hidden layer.
        init: str, optional
          Weight initialization for filters.
        activation: str, optional
          Activation function applied.
        dropout: float, optional
          Dropout probability in the hidden layer(s).
        """
    super(DAGGather, self).__init__(**kwargs)
    self.init = init
    self.activation = activation
    self.layer_sizes = layer_sizes
    self.dropout = dropout
    self.max_atoms = max_atoms
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs

  def _build_layer(self):
    return deepchem.models.layers.DAGGather(
        self.n_graph_feat, self.n_outputs, self.max_atoms, self.layer_sizes,
        self.init, self.activation, self.dropout)


class MessagePassing(KerasLayer):
  """ General class for MPNN
  default structures built according to https://arxiv.org/abs/1511.06391 """

  def __init__(self,
               T,
               message_fn='enn',
               update_fn='gru',
               n_hidden=100,
               **kwargs):
    """
    Parameters
    ----------
    T: int
      Number of message passing steps
    message_fn: str, optional
      message function in the model
    update_fn: str, optional
      update function in the model
    n_hidden: int, optional
      number of hidden units in the passing phase
    """

    self.T = T
    self.message_fn = message_fn
    self.update_fn = update_fn
    self.n_hidden = n_hidden
    super(MessagePassing, self).__init__(**kwargs)

  def _build_layer(self):
    return deepchem.models.layers.MessagePassing(self.T, self.message_fn,
                                                 self.update_fn, self.n_hidden)


class SetGather(KerasLayer):
  """ set2set gather layer for graph-based model
  model using this layer must set pad_batches=True """

  def __init__(self, M, batch_size, n_hidden=100, init='orthogonal', **kwargs):
    """
    Parameters
    ----------
    M: int
      Number of LSTM steps
    batch_size: int
      Number of samples in a batch(all batches must have same size)
    n_hidden: int, optional
      number of hidden units in the passing phase
    """
    self.M = M
    self.batch_size = batch_size
    self.n_hidden = n_hidden
    self.init = init
    super(SetGather, self).__init__(**kwargs)

  def _build_layer(self):
    return deepchem.models.layers.SetGather(self.M, self.batch_size,
                                            self.n_hidden, self.init)
