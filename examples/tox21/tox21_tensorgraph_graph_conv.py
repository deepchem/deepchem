"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np

from deepchem.models.tensorgraph.models.graph_conv import GraphConvTensorGraph
from models.tensorgraph.layers import Input, GraphConvLayer, BatchNormLayer, GraphPoolLayer, Dense, GraphGather, \
  SoftMax, SoftMaxCrossEntropy, Concat, WeightedError

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from tox21_datasets import load_tox21


def graph_conv_model(batch_size, num_tasks):
  model = GraphConvTensorGraph(batch_size=batch_size, use_queue=False)
  atom_features = Input(shape=(None, 75))
  model.add_feature(atom_features)

  degree_slice = Input(shape=(None, 2), dtype=tf.int32)
  model.add_feature(degree_slice)

  membership = Input(shape=(None,), dtype=tf.int32)
  model.add_feature(membership)

  deg_adjs = []
  for i in range(model.min_degree, model.max_degree + 1):
    deg_adj = Input(shape=(None, i + 1), dtype=tf.int32)
    model.add_feature(deg_adj)
    deg_adjs.append(deg_adj)

  gc1 = GraphConvLayer(64, activation_fn=tf.nn.relu)
  model.add_layer(
      gc1, parents=[atom_features, degree_slice, membership] + deg_adjs)

  batch_norm1 = BatchNormLayer()
  model.add_layer(batch_norm1, parents=[gc1])

  gp1 = GraphPoolLayer()
  model.add_layer(
      gp1, parents=[batch_norm1, degree_slice, membership] + deg_adjs)

  gc2 = GraphConvLayer(64, activation_fn=tf.nn.relu)
  model.add_layer(gc2, parents=[gp1, degree_slice, membership] + deg_adjs)

  batch_norm2 = BatchNormLayer()
  model.add_layer(batch_norm2, parents=[gc2])

  gp2 = GraphPoolLayer()
  model.add_layer(
      gp2, parents=[batch_norm2, degree_slice, membership] + deg_adjs)

  dense = Dense(out_channels=128, activation_fn=None)
  model.add_layer(dense, parents=[gp2])

  batch_norm3 = BatchNormLayer()
  model.add_layer(batch_norm3, parents=[dense])

  gg1 = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)
  model.add_layer(
      gg1, parents=[batch_norm3, degree_slice, membership] + deg_adjs)

  costs = []
  for task in range(num_tasks):
    classification = Dense(
        out_channels=2, name="GUESS%s" % task, activation_fn=None)
    model.add_layer(classification, parents=[gg1])

    softmax = SoftMax(name="SOFTMAX%s" % task)
    model.add_layer(softmax, parents=[classification])
    model.add_output(softmax)

    label = Input(shape=(None, 2), name="LABEL%s" % task)
    model.add_label(label)

    cost = SoftMaxCrossEntropy(name="COST%s" % task)
    model.add_layer(cost, parents=[label, classification])
    costs.append(cost)

  entropy = Concat(name="ENT")
  model.add_layer(entropy, parents=costs)

  task_weights = Input(shape=(None, num_tasks), name="W")
  model.add_task_weight(task_weights)

  loss = WeightedError(name="ERROR")
  model.add_layer(loss, parents=[entropy, task_weights])
  model.set_loss(loss)
  return model


# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# Batch size of models
batch_size = 50

model = graph_conv_model(batch_size, len(tox21_tasks))

model.fit(train_dataset, nb_epoch=10, checkpoint_interval=10)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
