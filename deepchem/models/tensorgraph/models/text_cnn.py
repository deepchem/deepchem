"""
Created on Thu Sep 28 15:17:50 2017

@author: zqwu
"""
import numpy as np
import tensorflow as tf
import copy
import sys

from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.trans import undo_transforms
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Conv1D, Concatenate, Lambda

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


class TextCNNModel(KerasModel):
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

    # Build the model.

    smiles_seqs = Input(shape=(self.seq_length,), dtype=tf.int32)
    # Character embedding
    embedding = layers.DTNNEmbedding(
        n_embedding=self.n_embedding,
        periodic_table_length=len(self.char_dict.keys()) + 1)(smiles_seqs)
    pooled_outputs = []
    conv_layers = []
    for filter_size, num_filter in zip(self.kernel_sizes, self.num_filters):
      # Multiple convolutional layers with different filter widths
      conv_layers.append(
          Conv1D(kernel_size=filter_size, filters=num_filter,
                 padding='valid')(embedding))
      # Max-over-time pooling
      reduced = Lambda(lambda x: tf.reduce_max(x, axis=1))(conv_layers[-1])
      pooled_outputs.append(reduced)
    # Concat features from all filters(one feature per filter)
    concat_outputs = Concatenate(axis=1)(pooled_outputs)
    dropout = Dropout(rate=self.dropout)(concat_outputs)
    dense = Dense(200, activation=tf.nn.relu)(dropout)
    # Highway layer from https://arxiv.org/pdf/1505.00387.pdf
    gather = layers.Highway()(dense)

    if self.mode == "classification":
      logits = Dense(self.n_tasks * 2)(gather)
      logits = Reshape((self.n_tasks, 2))(logits)
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss = SoftmaxCrossEntropy()

    else:
      output = Dense(self.n_tasks * 1)(gather)
      output = Reshape((self.n_tasks, 1))(output)
      outputs = [output]
      output_types = ['prediction']
      loss = L2Loss()

    model = tf.keras.Model(inputs=[smiles_seqs], outputs=outputs)
    super(TextCNNModel, self).__init__(
        model, loss, output_types=output_types, **kwargs)

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
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    """Transfer smiles strings to fixed length integer vectors"""
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if y_b is not None:
          if self.mode == 'classification':
            y_b = to_one_hot(y_b.flatten(), 2).reshape(-1, self.n_tasks, 2)
        # Transform SMILES sequence to integers
        X_b = self.smiles_to_seq_batch(ids_b)
        yield ([X_b], [y_b], [w_b])

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


#################### Deprecation warnings for renamed TensorGraph models ####################

import warnings

TENSORGRAPH_DEPRECATION = "{} is deprecated and has been renamed to {} and will be removed in DeepChem 3.0."


class TextCNNTensorGraph(TextCNNModel):

  def __init__(self, *args, **kwargs):
    warnings.warn(
        TENSORGRAPH_DEPRECATION.format("TextCNNTensorGraph", "TextCNNModel"),
        FutureWarning)

    super(TextCNNTensorGraph, self).__init__(*args, **kwargs)
