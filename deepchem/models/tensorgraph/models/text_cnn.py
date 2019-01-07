"""
Created on Thu Sep 28 15:17:50 2017

@author: zqwu
"""
import numpy as np
import tensorflow as tf
import copy
import sys

from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.tensorgraph.layers import Dense, Concat, SoftMax, \
  SoftMaxCrossEntropy, BatchNorm, WeightedError, Dropout, \
  Conv1D, ReduceMax, Squeeze, Stack, Highway
from deepchem.models.tensorgraph.graph_layers import DTNNEmbedding

from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature, Reshape, ReduceSum
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms

# Common symbols in SMILES, note that Cl and Br are regarded as single symbol
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
    's': 33
}


class TextCNNModel(TensorGraph):
  """ A Convolutional neural network on smiles strings
  Reimplementation of the discriminator module in ORGAN: https://arxiv.org/abs/1705.10843
  Originated from: http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf

  This model applies multiple 1D convolutional filters to the padded strings,
  then max-over-time pooling is applied on all filters, extracting one feature per filter.
  All features are concatenated and transformed through several hidden layers to form predictions.

  This model is initially developed for sentence-level classification tasks, with
  words represented as vectors. In this implementation, SMILES strings are dissected
  into characters and transformed to one-hot vectors in a similar way. The model can
  be used for general molecular-level classification or regression tasks. It is also
  used in the ORGAN model as discriminator.

  Training of the model only requires SMILES strings input, all featurized datasets
  that include SMILES in the `ids` attribute are accepted. PDBbind, QM7 and QM7b
  are not supported. To use the model, `build_char_dict` should be called first
  before defining the model to build character dict of input dataset, example can
  be found in examples/delaney/delaney_textcnn.py

  """

  def __init__(
      self,
      n_tasks,
      char_dict,
      seq_length,
      n_embedding=75,
      kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
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
      Mapping from characters in smiles to integers
    seq_length: int
      Length of sequences(after padding)
    n_embedding: int, optional
      Length of embedding vector
    filter_sizes: list of int, optional
      Properties of filters used in the conv net
    num_filters: list of int, optional
      Properties of filters used in the conv net
    dropout: float, optional
      Dropout rate
    mode: str
      Either "classification" or "regression" for type of model.
    """
    self.n_tasks = n_tasks
    self.char_dict = char_dict
    self.seq_length = max(seq_length, max(kernel_sizes))
    self.n_embedding = n_embedding
    self.kernel_sizes = kernel_sizes
    self.num_filters = num_filters
    self.dropout = dropout
    self.mode = mode
    super(TextCNNModel, self).__init__(**kwargs)
    self._build_graph()

  @staticmethod
  def build_char_dict(dataset, default_dict=default_dict):
    """ Collect all unique characters(in smiles) from the dataset.
    This method should be called before defining the model to build appropriate char_dict
    """
    # SMILES strings
    X = dataset.ids
    # Maximum length is expanded to allow length variation during train and inference
    seq_length = int(max([len(smile) for smile in X]) * 1.2)
    # '_' served as delimiter and padding
    all_smiles = '_'.join(X)
    tot_len = len(all_smiles)
    # Initialize common characters as keys
    keys = list(default_dict.keys())
    out_dict = copy.deepcopy(default_dict)
    current_key_val = len(keys) + 1
    # Include space to avoid extra keys
    keys.extend([' '])
    extra_keys = []
    i = 0
    while i < tot_len:
      # For 'Cl', 'Br', etc.
      if all_smiles[i:i + 2] in keys:
        i = i + 2
      elif all_smiles[i:i + 1] in keys:
        i = i + 1
      else:
        # Character not recognized, add to extra_keys
        extra_keys.append(all_smiles[i])
        keys.append(all_smiles[i])
        i = i + 1
    # Add all extra_keys to char_dict
    for extra_key in extra_keys:
      out_dict[extra_key] = current_key_val
      current_key_val += 1
    return out_dict, seq_length

  def _build_graph(self):
    self.smiles_seqs = Feature(shape=(None, self.seq_length), dtype=tf.int32)
    # Character embedding
    Embedding = DTNNEmbedding(
        n_embedding=self.n_embedding,
        periodic_table_length=len(self.char_dict.keys()) + 1,
        in_layers=[self.smiles_seqs])
    pooled_outputs = []
    conv_layers = []
    for filter_size, num_filter in zip(self.kernel_sizes, self.num_filters):
      # Multiple convolutional layers with different filter widths
      conv_layers.append(
          Conv1D(
              kernel_size=filter_size,
              filters=num_filter,
              padding='valid',
              in_layers=[Embedding]))
      # Max-over-time pooling
      pooled_outputs.append(ReduceMax(axis=1, in_layers=[conv_layers[-1]]))
    # Concat features from all filters(one feature per filter)
    concat_outputs = Concat(axis=1, in_layers=pooled_outputs)
    dropout = Dropout(dropout_prob=self.dropout, in_layers=[concat_outputs])
    dense = Dense(
        out_channels=200, activation_fn=tf.nn.relu, in_layers=[dropout])
    # Highway layer from https://arxiv.org/pdf/1505.00387.pdf
    gather = Highway(in_layers=[dense])

    if self.mode == "classification":
      logits = Dense(
          out_channels=self.n_tasks * 2, activation_fn=None, in_layers=[gather])
      logits = Reshape(shape=(-1, self.n_tasks, 2), in_layers=[logits])
      output = SoftMax(in_layers=[logits])
      self.add_output(output)
      labels = Label(shape=(None, self.n_tasks, 2))
      loss = SoftMaxCrossEntropy(in_layers=[labels, logits])

    else:
      vals = Dense(
          out_channels=self.n_tasks * 1, activation_fn=None, in_layers=[gather])
      vals = Reshape(shape=(-1, self.n_tasks, 1), in_layers=[vals])
      self.add_output(vals)
      labels = Label(shape=(None, self.n_tasks, 1))
      loss = ReduceSum(L2Loss(in_layers=[labels, vals]))

    weights = Weights(shape=(None, self.n_tasks))
    weighted_loss = WeightedError(in_layers=[loss, weights])
    self.set_loss(weighted_loss)

  @staticmethod
  def convert_bytes_to_char(s):
    s = ''.join(chr(b) for b in s)
    return s

  def smiles_to_seq_batch(self, ids_b):
    """Converts SMILES strings to np.array sequence.

    A tf.py_func wrapper is written around this when creating the input_fn for make_estimator
    """
    if isinstance(
        ids_b[0], bytes
    ) and sys.version_info[0] != 2:  # Python 2.7 bytes and string are analogous
      ids_b = [TextCNNModel.convert_bytes_to_char(smiles) for smiles in ids_b]
    smiles_seqs = [self.smiles_to_seq(smiles) for smiles in ids_b]
    smiles_seqs = np.vstack(smiles_seqs)
    return smiles_seqs

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    """ Transfer smiles strings to fixed length integer vectors
    """
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          if self.mode == "classification":
            feed_dict[self.labels[0]] = to_one_hot(y_b.flatten(), 2).reshape(
                -1, self.n_tasks, 2)
          else:
            feed_dict[self.labels[0]] = y_b
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b

        # Transform SMILES sequence to integers
        feed_dict[self.smiles_seqs] = self.smiles_to_seq_batch(ids_b)
        yield feed_dict

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    """Creates tensors for inputs."""
    tensors = dict()
    for layer, column in zip(self.features, feature_columns):
      feature_col = tf.feature_column.input_layer(features, [column])
      if column.dtype != feature_col.dtype:
        feature_col = tf.cast(feature_col, column.dtype)
      if len(column.shape) < 1:
        feature_col = tf.reshape(feature_col, shape=[tf.shape(feature_col)[0]])
      tensors[layer] = feature_col
    if weight_column is not None:
      tensors[self.task_weights[0]] = tf.feature_column.input_layer(
          features, [weight_column])
    if labels is not None:
      if self.mode == "classification":
        tensors[self.labels[0]] = tf.one_hot(tf.cast(labels, tf.int32), 2)
      else:
        tensors[self.labels[0]] = labels
    return tensors

  def smiles_to_seq(self, smiles):
    """ Tokenize characters in smiles to integers
    """
    smiles_len = len(smiles)
    seq = [0]
    keys = self.char_dict.keys()
    i = 0
    while i < smiles_len:
      # Skip all spaces
      if smiles[i:i + 1] == ' ':
        i = i + 1
      # For 'Cl', 'Br', etc.
      elif smiles[i:i + 2] in keys:
        seq.append(self.char_dict[smiles[i:i + 2]])
        i = i + 2
      elif smiles[i:i + 1] in keys:
        seq.append(self.char_dict[smiles[i:i + 1]])
        i = i + 1
      else:
        raise ValueError('character not found in dict')
    for i in range(self.seq_length - len(seq)):
      # Padding with '_'
      seq.append(self.char_dict['_'])
    return np.array(seq, dtype=np.int32)

  def predict_on_generator(self, generator, transformers=[], outputs=None):
    out = super(TextCNNModel, self).predict_on_generator(
        generator, transformers=[], outputs=outputs)
    if outputs is None:
      outputs = self.outputs
    if len(outputs) > 1:
      out = np.stack(out, axis=1)

    out = undo_transforms(out, transformers)
    return out


#################### Deprecation warnings for renamed TensorGraph models ####################

import warnings

TENSORGRAPH_DEPRECATION = "{} is deprecated and has been renamed to {} and will be removed in DeepChem 3.0."


class TextCNNTensorGraph(TextCNNModel):

  def __init__(self, *args, **kwargs):
    warnings.warn(
        TENSORGRAPH_DEPRECATION.format("TextCNNTensorGraph", "TextCNNModel"),
        FutureWarning)

    super(TextCNNTensorGraph, self).__init__(*args, **kwargs)
