"""
Smiles2Vec model, described in https://arxiv.org/pdf/1712.02034.pdf
ChemCeption model, described in https://arxiv.org/pdf/1706.06689.pdf
"""

from __future__ import division
from __future__ import unicode_literals

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import tensorflow as tf
import os
import sys
import logging

from deepchem.data.datasets import pad_batch
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.metrics import to_one_hot
from deepchem.models.tensorgraph.layers import KerasLayer
from deepchem.models.tensorgraph import chemnet_layers
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
from tensorflow.keras.layers import Dropout, Conv1D, Concatenate, Lambda, GRU, LSTM, Bidirectional
from tensorflow.keras.layers import Conv2D, ReLU, Add, GlobalAveragePooling2D

DEFAULT_INCEPTION_BLOCKS = {"A": 3, "B": 3, "C": 3}

INCEPTION_DICT = {
    "A": chemnet_layers.InceptionResnetA,
    "B": chemnet_layers.InceptionResnetB,
    "C": chemnet_layers.InceptionResnetC
}

RNN_DICT = {"GRU": GRU, "LSTM": LSTM}


class Smiles2Vec(KerasModel):
  """
  Smiles2Vec is a RNN basel model that ingests SMILES characters,
  get their embeddings. These embeddings are fed through a series of RNN
  (and optionally Conv) layers, with the representations from the final layer
  as input for classification or regression tasks.
  """

  def __init__(self,
               char_to_idx,
               n_tasks=10,
               batch_size=64,
               max_seq_len=270,
               embedding_dim=50,
               n_classes=2,
               use_bidir=True,
               use_conv=True,
               filters=192,
               kernel_size=3,
               strides=1,
               rnn_sizes=[224, 384],
               rnn_types=["GRU", "GRU"],
               learning_rate=0.0001,
               model_dir=None,
               tensorboard=True,
               optimizer=None,
               mode="regression",
               tensorboard_log_frequency=5,
               restore_from=None,
               **kwargs):
    """
    Parameters
    ----------
    char_to_idx: dict,
        char_to_idx contains character to index mapping for SMILES characters
    embedding_dim: int, default 50
        Size of character embeddings used.
    use_bidir: bool, default True
        Whether to use BiDirectional RNN Cells
    use_conv: bool, default True
        Whether to use a conv-layer
    kernel_size: int, default 3
        Kernel size for convolutions
    filters: int, default 192
        Number of filters
    strides: int, default 1
        Strides used in convolution
    rnn_sizes: list[int], default [224, 384]
        Number of hidden units in the RNN cells
    learning_rate: floatm default 0.0001,
        Default learning rate used for the model
    model_dir: str, default None
        Directory to save model to
    tensorboard: bool, default True
        Whether to use TensorBoard
    optimizer: tf.train.Optimizer, default None
        Optimizer used.
    mode: str, default regression
        Whether to use model for regression or classification
    tensorboard_log_frequency: int, default 5
        How often to log to TensorBoard
    restore_from: str, default None
        Where to restore model from
    """

    self.char_to_idx = char_to_idx
    self.n_classes = n_classes
    self.max_seq_len = max_seq_len
    self.embedding_dim = embedding_dim
    self.use_bidir = use_bidir
    self.use_conv = use_conv
    if use_conv:
      self.kernel_size = kernel_size
      self.filters = filters
      self.strides = strides
    self.rnn_types = rnn_types
    self.rnn_sizes = rnn_sizes
    assert len(rnn_sizes) == len(
        rnn_types), "Should have same number of hidden units as RNNs"
    self.n_tasks = n_tasks
    self.mode = mode
    self.batch_size = batch_size

    model, loss, output_types = self._build_graph()
    super(Smiles2Vec, self).__init__(
        model=model,
        loss=loss,
        output_types=output_types,
        model_dir=model_dir,
        tensorboard=tensorboard,
        batch_size=batch_size,
        optimizer=optimizer,
        tensorboard_log_frequency=tensorboard_log_frequency,
        learning_rate=learning_rate,
        **kwargs)

  def _build_graph(self):
    """Build the model."""
    smiles_seqs = Input(dtype=tf.int32, shape=(self.max_seq_len,), name='Input')
    rnn_input = chemnet_layers.SmilesEmbedding(
        charset_size=len(self.char_to_idx),
        embedding_dim=self.embedding_dim,
        name='Embedding')(smiles_seqs)

    if self.use_conv:
      rnn_input = Conv1D(
          filters=self.filters,
          kernel_size=self.kernel_size,
          strides=self.strides,
          activation=tf.nn.relu,
          name='Conv1D')(rnn_input)

    rnn_embeddings = rnn_input
    for idx, rnn_type in enumerate(self.rnn_types[:-1]):
      rnn_layer = RNN_DICT[rnn_type]
      layer = rnn_layer(units=self.rnn_sizes[idx], return_sequences=True)
      if self.use_bidir:
        layer = Bidirectional(layer)

      rnn_embeddings = layer(rnn_embeddings)

    # Last layer sequences not returned.
    layer = RNN_DICT[self.rnn_types[-1]](units=self.rnn_sizes[-1])
    if self.use_bidir:
      layer = Bidirectional(layer)
    rnn_embeddings = layer(rnn_embeddings)

    if self.mode == "classification":
      logits = Dense(self.n_tasks * 2)(rnn_embeddings)
      logits = Reshape((self.n_tasks, 2))(logits)
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss = SoftmaxCrossEntropy()

    else:
      output = Dense(self.n_tasks * 1, name='Dense')(rnn_embeddings)
      output = Reshape((self.n_tasks, 1), name='Reshape')(output)
      outputs = [output]
      output_types = ['prediction']
      loss = L2Loss()

    model = tf.keras.Model(inputs=[smiles_seqs], outputs=outputs)
    return model, loss, output_types

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if self.mode == 'classification':
          y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
              -1, self.n_tasks, self.n_classes)
        yield ([X_b], [y_b], [w_b])


class ChemCeption(KerasModel):
  """
  ChemCeption is a CNN based model composed of a series of Inception-ResNet and
  Reduction layers. The representations from the final layer serve as input for
  classification or regression tasks. The input for the ChemCeption model is an
  image representation of the molecule, described in SmilesToImage featurizer.
  """

  def __init__(self,
               img_spec="std",
               base_filters=16,
               inception_blocks=DEFAULT_INCEPTION_BLOCKS,
               n_tasks=10,
               n_classes=2,
               batch_size=100,
               learning_rate=0.001,
               augment=False,
               tensorboard=True,
               tensorboard_log_frequency=5,
               model_dir=None,
               optimizer=None,
               mode="regression",
               **kwargs):
    """
    Parameters
    ----------
    img_spec: str, default std
        Image specification used
    base_filters: int, default 16
        Base filters used for the different inception and reduction layers
    inception_blocks: dict,
        Dictionary containing number of blocks for every inception layer
    n_tasks: int, default 10
        Number of classification or regression tasks
    n_classes: int, default 2
        Number of classes (used only for classification)
    batch_size: int, default 100
        Minibatch size used for model fitting
    learning_rate: float, default 0.0001
        Learning rate used for training
    augment: bool, default False
        Whether to augment images
    tensorboard: bool, default True
        Whether to log to TensorBoard (does not work in eager mode.)
    tensorboard_log_frequency: int, default 5
        Frequency to log to TensorBoard
    model_dir: str, default None,
        Directory to save the model in
    optimizer: tf.train.Optimizer, default None
        Optimizer used for training
    mode: str, default regression
        Whether the model is used for regression or classification
    """
    if img_spec == "engd":
      self.input_shape = (80, 80, 4)
    elif img_spec == "std":
      self.input_shape = (80, 80, 1)
    self.base_filters = base_filters
    self.inception_blocks = inception_blocks
    self.n_tasks = n_tasks
    self.n_classes = n_classes
    self.mode = mode
    self.augment = augment

    model, loss, output_types = self._build_graph()
    super(ChemCeption, self).__init__(
        model=model,
        loss=loss,
        output_types=output_types,
        model_dir=model_dir,
        tensorboard=tensorboard,
        batch_size=batch_size,
        optimizer=optimizer,
        tensorboard_log_frequency=tensorboard_log_frequency,
        learning_rate=learning_rate,
        **kwargs)

  def _build_graph(self):
    smile_images = Input(shape=self.input_shape)
    stem = chemnet_layers.Stem(self.base_filters)(smile_images)

    inceptionA_out = self.build_inception_module(inputs=stem, type="A")
    reductionA_out = chemnet_layers.ReductionA(
        self.base_filters)(inceptionA_out)

    inceptionB_out = self.build_inception_module(
        inputs=reductionA_out, type="B")
    reductionB_out = chemnet_layers.ReductionB(
        self.base_filters)(inceptionB_out)

    inceptionC_out = self.build_inception_module(
        inputs=reductionB_out, type="C")
    avg_pooling_out = GlobalAveragePooling2D()(inceptionC_out)

    if self.mode == "classification":
      logits = Dense(self.n_tasks * 2)(avg_pooling_out)
      logits = Reshape((self.n_tasks, 2))(logits)
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss = SoftmaxCrossEntropy()

    else:
      output = Dense(self.n_tasks * 1)(avg_pooling_out)
      output = Reshape((self.n_tasks, 1))(output)
      outputs = [output]
      output_types = ['prediction']
      loss = L2Loss()

    model = tf.keras.Model(inputs=[smile_images], outputs=outputs)
    return model, loss, output_types

  def build_inception_module(self, inputs, type="A"):
    """Inception module is a series of inception layers of similar type. This
    function builds that."""
    num_blocks = self.inception_blocks[type]
    inception_layer = INCEPTION_DICT[type]
    output = inputs
    for block in range(num_blocks):
      output = inception_layer(self.base_filters, int(inputs.shape[-1]))(output)
    return output

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      if mode == "predict" or (not self.augment):
        for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
            batch_size=self.batch_size,
            deterministic=deterministic,
            pad_batches=pad_batches):
          if self.mode == 'classification':
            y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                -1, self.n_tasks, self.n_classes)
          yield ([X_b], [y_b], [w_b])

      else:
        if not pad_batches:
          n_samples = dataset.X.shape[0]
        else:
          n_samples = dataset.X.shape[0] + (
              self.batch_size - (dataset.X.shape[0] % self.batch_size))

        n_batches = 0
        image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=180)
        for (X_b, y_b, w_b) in image_data_generator.flow(
            dataset.X,
            dataset.y,
            sample_weight=dataset.w,
            shuffle=not deterministic,
            batch_size=self.batch_size):
          if pad_batches:
            ids_b = np.arange(X_b.shape[0])
            X_b, y_b, w_b, _ = pad_batch(self.batch_size, X_b, y_b, w_b, ids_b)
          n_batches += 1
          if n_batches > n_samples / self.batch_size:
            # This is needed because ImageDataGenerator does infinite looping
            break
          if self.mode == "classification":
            y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                -1, self.n_tasks, self.n_classes)
          yield ([X_b], [y_b], [w_b])
