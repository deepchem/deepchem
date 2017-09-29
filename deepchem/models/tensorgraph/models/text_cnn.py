#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:17:50 2017

@author: zqwu
"""
import numpy as np
import tensorflow as tf
import copy

from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.tensorgraph.layers import Dense, Concat, SoftMax, \
  SoftMaxCrossEntropy, BatchNorm, WeightedError, Dropout, BatchNormalization, \
  Conv1D, MaxPool1D, Squeeze, Stack
from deepchem.models.tensorgraph.graph_layers import DTNNEmbedding

from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms

default_dict = {
 '#': 1,
 '(': 2,
 ')': 3,
 '+': 4,
 '-': 5,
 '/': 6,
 '1': 7,
 '2': 8,
 '3': 9,
 '4': 10,
 '5': 11,
 '6': 12,
 '7': 13,
 '8': 14,
 '=': 15,
 'C': 16,
 'F': 17,
 'H': 18,
 'I': 19,
 'N': 20,
 'O': 21,
 'P': 22,
 'S': 23,
 '[': 24,
 '\\': 25,
 ']': 26,
 '_': 27,
 'c': 28,
 'Cl': 29,
 'Br': 30,
 'n': 31,
 'o': 32,
 's': 33}

class TextCNNTensorGraph(TensorGraph):

  def __init__(self,
               n_tasks,
               char_dict,
               seq_length,
               n_embedding=75,
               filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
               num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
               dropout=0.25,
               mode="classification",
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    char_dict: dict
      Mapping from characters in smiles to integer tokens
    seq_length: int
      Length of sequences(after padding)
    n_embedding: int, optional
      Length of embedding vector
    filter_sizes: list of int, optional
      Properties of filters used in the conv net
    num_filters: list of int, optional
      Properties of filters used in the conv net
    mode: str
      Either "classification" or "regression" for type of model.
    """
    self.n_tasks = n_tasks
    self.char_dict = char_dict
    self.seq_length = seq_length
    self.n_embedding = n_embedding
    self.filter_sizes = filter_sizes
    self.num_filters = num_filters
    self.dropout = dropout
    self.mode = mode
    super(TextCNNTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  @staticmethod
  def build_char_dict(dataset, default_dict=default_dict):
    X = dataset.ids
    seq_length = int(max([len(smile) for smile in X]) * 1.2)
    all_smiles = '_'.join(X)
    tot_len = len(all_smiles)
    keys = default_dict.keys()
    out_dict = copy.deepcopy(default_dict)
    current_key_val = len(keys) + 1
    keys.extend(['_', ' '])
    extra_keys = []
    i = 0
    while i < tot_len:
      if all_smiles[i:i+2] in keys:
        i = i + 2
      elif all_smiles[i:i+1] in keys:
        i = i + 1
      else:
        extra_keys.append(all_smiles[i])
        keys.append(all_smiles[i])
        i = i + 1
    for extra_key in extra_keys:
      out_dict[extra_key] = current_key_val
      current_key_val += 1
    out_dict['_'] = current_key_val
    return out_dict, seq_length

  def build_graph(self):
    self.smiles_seqs = Feature(shape=(None, self.seq_length), dtype=tf.int32)
    self.Embedding = DTNNEmbedding(n_embedding=self.n_embedding,
                              periodic_table_length=len(self.char_dict.keys())+1,
                              in_layers=[self.smiles_seqs])
    self.pooled_outputs = []
    self.conv_layers = []
    for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
      self.conv_layers.append(Conv1D(filter_size,
                                     num_filter,
                                     padding='VALID',
                                     in_layers=[self.Embedding]))
      self.pooled_outputs.append(MaxPool1D(window_shape=self.seq_length-filter_size+1,
                                           strides=1,
                                           padding='VALID',
                                           in_layers=[self.conv_layers[-1]]))

    concat_outputs = Concat(axis=2, in_layers=self.pooled_outputs)
    outputs = Squeeze(squeeze_dims=1, in_layers=concat_outputs)
    #HIGHWAY LAYER
    highway = Highway(in_layers=[outputs])
    self.gather = Dropout(dropout_prob=self.dropout, in_layers=[highway])

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      if self.mode == "classification":
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[self.gather])
        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.labels_fd.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == "regression":
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[self.gather])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.labels_fd.append(label)
        cost = L2Loss(in_layers=[label, regression])
        costs.append(cost)
    if self.mode == "classification":
      all_cost = Concat(in_layers=costs, axis=1)
    elif self.mode == "regression":
      all_cost = Stack(in_layers=costs, axis=1)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)


  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    """ TensorGraph style implementation
    similar to deepchem.models.tf_new_models.graph_topology.AlternateWeaveTopology.batch_to_feed_dict
    """
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            if self.mode == "classification":
              feed_dict[label] = to_one_hot(y_b[:, index])
            if self.mode == "regression":
              feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None:
          feed_dict[self.weights] = w_b

        smiles_seqs = [self.smiles_to_seq(smiles) for smiles in ids_b]
        feed_dict[self.smiles_seqs] = np.stack(smiles_seqs, axis=0)
        yield feed_dict

  def smiles_to_seq(self, smiles):
    smiles_len = len(smiles)
    # Starting token
    seq = [0]
    keys = self.char_dict.keys()
    i = 0
    while i < smiles_len:
      if smiles[i:i+1] == ' ':
        i = i + 1
      if smiles[i:i+2] in keys:
        seq.append(self.char_dict[smiles[i:i+2]])
        i = i + 2
      elif smiles[i:i+1] in keys:
        seq.append(self.char_dict[smiles[i:i+1]])
        i = i + 1
      else:
        raise ValueError('character not found in dict')
    for i in range(self.seq_length - len(seq)):
      seq.append(self.char_dict['_'])
    return np.array(seq)