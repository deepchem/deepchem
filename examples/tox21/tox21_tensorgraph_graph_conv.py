"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import six

from deepchem.models.tensorgraph import TensorGraph
from metrics import to_one_hot

from feat.mol_graphs import ConvMol
from models.tensorgraph.layers import Input, GraphConvLayer, BatchNormLayer, GraphPoolLayer, Dense, GraphGather, \
  SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Weights, Feature

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from tox21_datasets import load_tox21

model_dir = "/tmp/graph_conv"


def graph_conv_model(batch_size, tasks):
  model = TensorGraph(
      model_dir=model_dir, batch_size=batch_size, use_queue=False)
  atom_features = Feature(shape=(None, 75))
  degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
  membership = Feature(shape=(None,), dtype=tf.int32)

  deg_adjs = []
  for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)
  gc1 = GraphConvLayer(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[atom_features, degree_slice, membership] + deg_adjs)
  batch_norm1 = BatchNormLayer(in_layers=[gc1])
  gp1 = GraphPoolLayer(
      in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)
  gc2 = GraphConvLayer(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[gp1, degree_slice, membership] + deg_adjs)
  batch_norm2 = BatchNormLayer(in_layers=[gc2])
  gp2 = GraphPoolLayer(
      in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)
  dense = Dense(out_channels=128, activation_fn=None, in_layers=[gp2])
  batch_norm3 = BatchNormLayer(in_layers=[dense])
  gg1 = GraphGather(
      batch_size=batch_size,
      activation_fn=tf.nn.tanh,
      in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)

  costs = []
  labels = []
  for task in tasks:
    classification = Dense(out_channels=2, activation_fn=None, in_layers=[gg1])

    softmax = SoftMax(in_layers=[classification])
    model.add_output(softmax)

    label = Label(shape=(None, 2))
    labels.append(label)
    cost = SoftMaxCrossEntropy(in_layers=[label, classification])
    costs.append(cost)

  entropy = Concat(in_layers=costs)
  task_weights = Weights(shape=(None, len(tasks)))
  loss = WeightedError(in_layers=[entropy, task_weights])
  model.set_loss(loss)

  def feed_dict_generator(dataset, batch_size, epochs=1):
    for epoch in range(epochs):
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(batch_size, pad_batches=True)):
        d = {}
        for index, label in enumerate(labels):
          d[label] = to_one_hot(y_b[:, index])
        d[task_weights] = w_b
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        d[atom_features] = multiConvMol.get_atom_features()
        d[degree_slice] = multiConvMol.deg_slice
        d[membership] = multiConvMol.membership
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
        yield d

  return model, feed_dict_generator, labels, task_weights


# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets
print(train_dataset.data_dir)
print(valid_dataset.data_dir)

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# Batch size of models
batch_size = 50

model, generator, labels, task_weights = graph_conv_model(batch_size,
                                                          tox21_tasks)

model.fit_generator(generator(train_dataset, batch_size, epochs=10))

print("Evaluating model")
train_scores = model.evaluate_generator(
    generator(train_dataset, batch_size), [metric],
    transformers,
    labels,
    weights=[task_weights])
valid_scores = model.evaluate_generator(
    generator(valid_dataset, batch_size), [metric],
    transformers,
    labels,
    weights=[task_weights])

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
