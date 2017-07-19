"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import six

from deepchem.models.tensorgraph import TensorGraph
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Input, GraphConv, Add, SluiceLoss, BatchNorm, GraphPool, Dense, GraphGather, BetaShare,  LayerSplitter, SoftMax, SoftMaxCrossEntropy, Concat, WeightedError, Label, Weights, Feature, AlphaShare

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from sluice_data import load_sluice


model_dir = "/tmp/graph_conv"


def to_one_hot(y, is_first):
    n_samples = np.shape(y)[0]
    n_classes = 0
    if is_first:
        n_classes = 20
    else:
        n_classes = 20
    y_hot = np.zeros((n_samples, n_classes), dtype='float64')
    for row, value in enumerate(y):
        y_hot[row, value] = 1
    return y_hot


def graph_conv_model(batch_size, tasks):
    model = TensorGraph(model_dir=model_dir,
                        batch_size=batch_size, use_queue=False)
    sluice_cost = []

    X1 = Feature(shape=(None, 10))
    X2 = Feature(shape=(None, 10))

    d1a = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X1])
    d1b = Dense(out_channels=10, activation_fn=tf.nn.relu, in_layers=[X2])

    d1c = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1a])
    d1d = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[d1b])

    sluice_cost.append(d1c)
    sluice_cost.append(d1d)

    as1 = AlphaShare(in_layers=[d1c, d1d])
    ls1a = LayerSplitter(in_layers=[as1], tower_num=0)
    ls1b = LayerSplitter(in_layers=[as1], tower_num=1)

    d2a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls1a])
    d2b = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls1b])

    sluice_cost.append(d2a)
    sluice_cost.append(d2b)

    as2 = AlphaShare(in_layers=[d2a, d2b])
    ls2a = LayerSplitter(in_layers=[as2], tower_num=0)
    ls2b = LayerSplitter(in_layers=[as2], tower_num=1)

    d3a = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls2a])
    d3b = Dense(out_channels=20, activation_fn=tf.nn.relu, in_layers=[ls2b])

    b1 = BetaShare(in_layers=[ls1a, ls2a])
    b2 = BetaShare(in_layers=[ls1b, ls2b])

    count = 0
    costs = []
    labels = []

    for task in tasks:
        if count < len(tasks) / 2:
            classification = Dense(
                out_channels=20, activation_fn=None, in_layers=[b1])
            label = Label(shape=(None, 20))
        else:
            classification = Dense(
                out_channels=20, activation_fn=None, in_layers=[b2])
            label = Label(shape=(None, 20))
        count += 1
        softmax = SoftMax(in_layers=[classification])
        model.add_output(softmax)

        labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)

    s_cost = SluiceLoss(in_layers=sluice_cost)
    entropy = Concat(in_layers=costs)
    task_weights = Weights(shape=(None, len(tasks)))
    loss = WeightedError(in_layers=[entropy, task_weights])
    loss = Add(in_layers=[loss, s_cost])
    model.set_alphas([as1,as2])
    model.set_betas([b1,b2])

    model.set_loss(loss)

    def feed_dict_generator(dataset, batch_size, epochs=1):
        for epoch in range(epochs):
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                    dataset.iterbatches(batch_size, pad_batches=True)):
                d = {}
                d[X1] = X_b
                d[X2] = X_b
                is_first = True
                for index, label in enumerate(labels):
                    d[label] = to_one_hot(y_b[:, index], is_first=is_first)
                    is_first = False
                d[task_weights] = w_b
                yield d
    return model, feed_dict_generator, labels, task_weights

# Load Tox21 dataset
sluice_tasks, sluice_datasets, transformers = load_sluice()
train_dataset, valid_dataset, test_dataset = sluice_datasets
print("train dataset shape:")
print(train_dataset.get_shape)
print("valid dataset shape")
print(valid_dataset.get_shape)
print("test dataset shape")
print(test_dataset.get_shape)

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.mean_absolute_error, np.mean, mode="classification")

# Batch size of models
batch_size = 100

model, generator, labels, task_weights = graph_conv_model(
    batch_size, sluice_tasks)

print('labels')
print(labels)

model.fit_generator(generator(train_dataset, batch_size, epochs=100))

print("Evaluating model")
train_scores = model.evaluate_generator(
    generator(train_dataset, batch_size), [metric],
    labels=labels, weights=[task_weights])
valid_scores = model.evaluate_generator(
    generator(valid_dataset, batch_size), [metric],
    labels=labels,
    weights=[task_weights])
test_scores = model.evaluate_generator(
    generator(test_dataset, batch_size), [metric],
    labels=labels,
    weights=[task_weights])

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

print('Test scores')
print(test_scores)
