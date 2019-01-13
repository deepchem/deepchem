"""DeepMHC model, found in https://www.biorxiv.org/content/early/2017/12/24/239236"""

from __future__ import division
from __future__ import unicode_literals

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import tensorflow as tf

from deepchem.data import NumpyDataset
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Conv1D, MaxPool1D, Dense, Dropout
from deepchem.models.tensorgraph.layers import Flatten
from deepchem.models.tensorgraph.layers import Feature, Weights, Label
from deepchem.models.tensorgraph.layers import L2Loss, WeightedError


class DeepMHC(TensorGraph):

  name = ['DeepMHC']

  def __init__(self,
               batch_size=64,
               pad_length=13,
               dropout_p=0.5,
               num_amino_acids=20,
               mode="regression",
               **kwargs):

    assert mode in ["regression", "classification"]
    self.mode = mode
    self.batch_size = batch_size
    self.dropout_p = dropout_p
    self.pad_length = pad_length
    self.num_amino_acids = num_amino_acids
    super(DeepMHC, self).__init__(**kwargs)
    self._build_graph()

  def _build_graph(self):

    self.one_hot_seq = Feature(
        shape=(None, self.pad_length, self.num_amino_acids), dtype=tf.float32)

    conv1 = Conv1D(kernel_size=2, filters=512, in_layers=[self.one_hot_seq])

    maxpool1 = MaxPool1D(strides=2, padding="VALID", in_layers=[conv1])
    conv2 = Conv1D(kernel_size=3, filters=512, in_layers=[maxpool1])
    flattened = Flatten(in_layers=[conv2])
    dense1 = Dense(
        out_channels=400, in_layers=[flattened], activation_fn=tf.nn.tanh)
    dropout = Dropout(dropout_prob=self.dropout_p, in_layers=[dense1])
    output = Dense(out_channels=1, in_layers=[dropout], activation_fn=None)
    self.add_output(output)

    if self.mode == "regression":
      label = Label(shape=(None, 1))
      loss = L2Loss(in_layers=[label, output])
    else:
      raise NotImplementedError(
          "Classification support not added yet. Missing details in paper.")
    weights = Weights(shape=(None,))
    weighted_loss = WeightedError(in_layers=[loss, weights])
    self.set_loss(weighted_loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):

    for epoch in range(epochs):
      for (X_b, y_b, w_b,
           ids_b) in dataset.iterbatches(batch_size=self.batch_size):
        feed_dict = {}
        feed_dict[self.one_hot_seq] = X_b
        if y_b is not None:
          feed_dict[self.labels[0]] = -np.log10(y_b)
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b

        yield feed_dict

  def predict_on_batch(self, X, transformers=[], outputs=None):
    dataset = NumpyDataset(X, y=None)
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    preds = self.predict_on_generator(generator, transformers, outputs)
    preds = 10**-preds  # Since we get train on -log10(IC50)
    return preds

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    tensors = dict()
    for layer, column in zip(self.features, feature_columns):
      feature_column = tf.feature_column.input_layer(features, [column])
      if feature_column.dtype != column.dtype:
        feature_column = tf.cast(feature_column, column.dtype)
      tensors[layer] = feature_column
    if weight_column is not None:
      tensors[self.task_weights[0]] = tf.feature_column.input_layer(
          features, [weight_column])
    if labels is not None:
      tensors[self.labels[[0]]] = labels

    return tensors
