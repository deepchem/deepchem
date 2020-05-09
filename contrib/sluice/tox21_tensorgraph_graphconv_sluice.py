"""
Script that trains graph-conv models on Tox21 dataset.
"""
import numpy as np
import sys

from deepchem.models.tensorgraph import TensorGraph
from deepchem.metrics import to_one_hot

from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Input, GraphConv, BatchNorm, GraphPool, Dense, GraphGather, \
  SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Constant, Weights, Feature, AlphaShare, SluiceLoss, Add

np.random.seed(123)
import tensorflow as tf

tf.random.set_seed(123)
import deepchem as dc
from deepchem.molnet import load_tox21


def sluice_model(batch_size, tasks):
  model = TensorGraph(
      model_dir=model_dir,
      batch_size=batch_size,
      use_queue=False,
      tensorboard=True)
  atom_features = Feature(shape=(None, 75))
  degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
  membership = Feature(shape=(None,), dtype=tf.int32)

  sluice_loss = []
  deg_adjs = []
  for i in range(0, 10 + 1):
    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
    deg_adjs.append(deg_adj)

  gc1 = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[atom_features, degree_slice, membership] + deg_adjs)

  as1 = AlphaShare(in_layers=[gc1, gc1])
  sluice_loss.append(gc1)

  batch_norm1a = BatchNorm(in_layers=[as1[0]])
  batch_norm1b = BatchNorm(in_layers=[as1[1]])

  gp1a = GraphPool(in_layers=[batch_norm1a, degree_slice, membership] +
                   deg_adjs)
  gp1b = GraphPool(in_layers=[batch_norm1b, degree_slice, membership] +
                   deg_adjs)

  gc2a = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[gp1a, degree_slice, membership] + deg_adjs)
  gc2b = GraphConv(
      64,
      activation_fn=tf.nn.relu,
      in_layers=[gp1b, degree_slice, membership] + deg_adjs)

  as2 = AlphaShare(in_layers=[gc2a, gc2b])
  sluice_loss.append(gc2a)
  sluice_loss.append(gc2b)

  batch_norm2a = BatchNorm(in_layers=[as2[0]])
  batch_norm2b = BatchNorm(in_layers=[as2[1]])

  gp2a = GraphPool(in_layers=[batch_norm2a, degree_slice, membership] +
                   deg_adjs)
  gp2b = GraphPool(in_layers=[batch_norm2b, degree_slice, membership] +
                   deg_adjs)

  densea = Dense(out_channels=128, activation_fn=None, in_layers=[gp2a])
  denseb = Dense(out_channels=128, activation_fn=None, in_layers=[gp2b])

  batch_norm3a = BatchNorm(in_layers=[densea])
  batch_norm3b = BatchNorm(in_layers=[denseb])

  as3 = AlphaShare(in_layers=[batch_norm3a, batch_norm3b])
  sluice_loss.append(batch_norm3a)
  sluice_loss.append(batch_norm3b)

  gg1a = GraphGather(
      batch_size=batch_size,
      activation_fn=tf.nn.tanh,
      in_layers=[as3[0], degree_slice, membership] + deg_adjs)
  gg1b = GraphGather(
      batch_size=batch_size,
      activation_fn=tf.nn.tanh,
      in_layers=[as3[1], degree_slice, membership] + deg_adjs)

  costs = []
  labels = []
  count = 0
  for task in tasks:
    if count < len(tasks) / 2:
      classification = Dense(
          out_channels=2, activation_fn=None, in_layers=[gg1a])
      print("first half:")
      print(task)
    else:
      classification = Dense(
          out_channels=2, activation_fn=None, in_layers=[gg1b])
      print('second half')
      print(task)
    count += 1

    softmax = SoftMax(in_layers=[classification])
    model.add_output(softmax)

    label = Label(shape=(None, 2))
    labels.append(label)
    cost = SoftMaxCrossEntropy(in_layers=[label, classification])
    costs.append(cost)

  entropy = Concat(in_layers=costs)
  task_weights = Weights(shape=(None, len(tasks)))
  task_loss = WeightedError(in_layers=[entropy, task_weights])

  s_cost = SluiceLoss(in_layers=sluice_loss)

  total_loss = Add(in_layers=[task_loss, s_cost])
  model.set_loss(total_loss)

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


model_dir = "tmp/graphconv"

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets
print(train_dataset.data_dir)
print(valid_dataset.data_dir)

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# Batch size of models
batch_size = 100

num_epochs = 10

model, generator, labels, task_weights = sluice_model(batch_size, tox21_tasks)

model.fit_generator(
    generator(train_dataset, batch_size, epochs=num_epochs),
    checkpoint_interval=1000)

print("Evaluating model")
train_scores = model.evaluate_generator(
    generator(train_dataset, batch_size), [metric],
    transformers,
    labels,
    weights=[task_weights],
    per_task_metrics=True)
valid_scores = model.evaluate_generator(
    generator(valid_dataset, batch_size), [metric],
    transformers,
    labels,
    weights=[task_weights],
    per_task_metrics=True)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
