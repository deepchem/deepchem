"""K-Hop Layer and Adaptive Filter Module from https://arxiv.org/pdf/1706.09916.pdf"""

from __future__ import division
from __future__ import unicode_literals

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import tensorflow as tf
import os
import sys
import logging

from deepchem.models.tensorgraph.layers import Layer, convert_to_layers
from deepchem.models.tensorgraph import initializations, model_ops
from deepchem.models.tensorgraph import activations


class AdaptiveFilter(Layer):

  def __init__(self,
               num_nodes,
               num_node_features,
               batch_size=64,
               init='glorot_uniform',
               combine_method='linear',
               **kwargs):
    """
      Parameters
      ----------
      num_nodes: int
        Number of nodes in the graph
      num_node_features: int
        Number of features per node in the graph
      batch_size: int, optional
        Batch size used for training
      init: str, optional
        Initialization method for the weights
      combine_method: str, optional
        How to combine adjacency matrix and node features

    """

    if combine_method not in ['linear', 'prod']:
      raise ValueError('Combine method needs to be one of linear or product')
    self.num_nodes = num_nodes
    self.num_node_features = num_node_features
    self.batch_size = batch_size
    self.init = initializations.get(init)
    self.combine_method = combine_method
    super(AdaptiveFilter, self).__init__(**kwargs)

  def _build(self):
    if self.combine_method == "linear":
      self.Q = self.init(
          shape=(self.num_nodes + self.num_node_features, self.num_nodes))
    else:
      self.Q = self.init(shape=(self.num_node_features, self.num_nodes))

    self.trainable_weights = [self.Q]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    act_fn = activations.get('sigmoid')
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    self._build()

    A_tilda_k = in_layers[0].out_tensor
    X = in_layers[1].out_tensor

    if self.combine_method == "linear":
      concatenated = tf.concat([A_tilda_k, X], axis=2)
      adp_fn_val = act_fn(
          tf.tensordot(concatenated, self.trainable_weights[0], axes=1))
    else:
      adp_fn_val = act_fn(tf.matmul(A_tilda_k, tf.tensordot(X, self.Q, axes=1)))
    out_tensor = adp_fn_val
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor

    return out_tensor

  def none_tensors(self):
    Q = self.Q
    self.Q = None
    out_tensor, trainable_weights, variables = self.out_tensor, self.trainable_weights, self.variables
    self.out_tensor, self.trainable_weights, self.variables = None, [], []
    return Q, out_tensor, trainable_weights, variables

  def set_tensors(self, tensors):
    self.Q, self.out_tensor, self.trainable_weights, self.variables = tensors


class KOrderGraphConv(Layer):

  name = ['KOrderGraphConv']

  def __init__(self,
               num_nodes,
               num_node_features,
               batch_size=64,
               init='glorot_uniform',
               **kwargs):
    """
      Parameters
      ----------
      num_nodes: int
        Number of nodes in the graph
      num_node_features: int
        Number of features per node in the graph
      batch_size: int, optional
        Batch size used for training
      init: str, optional
        Initialization method for the weights
      combine_method: str, optional
        How to combine adjacency matrix and node features

    """

    self.num_nodes = num_nodes
    self.num_node_features = num_node_features
    self.batch_size = batch_size
    self.init = initializations.get(init)

    super(KOrderGraphConv, self).__init__(**kwargs)

  def _build(self):
    self.W = self.init(shape=(self.num_nodes, self.num_nodes))
    self.b = model_ops.zeros(shape=[
        self.num_nodes,
    ])

    self.trainable_weights = [self.W, self.b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    self._build()

    A_tilda_k = in_layers[0].out_tensor
    X = in_layers[1].out_tensor
    adp_fn_val = in_layers[2].out_tensor

    attn_weights = tf.multiply(adp_fn_val, self.W)
    wt_adjacency = attn_weights * A_tilda_k
    out = tf.matmul(wt_adjacency, X) + tf.expand_dims(self.b, axis=1)

    out_tensor = out
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor

    return out_tensor

  def none_tensors(self):
    W, b = self.W, self.b
    self.W, self.b = None, None
    out_tensor, trainable_weights, variables = self.out_tensor, self.trainable_weights, self.variables
    self.out_tensor, self.trainable_weights, self.variables = None, [], []
    return W, b, out_tensor, trainable_weights, variables

  def set_tensors(self, tensors):
    self.W, self.b, self.out_tensor, self.trainable_weights, self.variables = tensors
