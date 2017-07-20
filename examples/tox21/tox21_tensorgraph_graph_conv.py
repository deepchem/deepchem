"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from models.tensorgraph import TFWrapper

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from tox21_datasets import load_tox21
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph

model_dir = "/tmp/graph_conv"

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

model = GraphConvTensorGraph(
    len(tox21_tasks), batch_size=batch_size, mode='classification')

global_step = model._get_tf('GlobalStep')


def optimizer_function():
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    return tf.train.GradientDescentOptimizer(learning_rate)


model.set_optimizer(TFWrapper(optimizer_function))

model.fit(train_dataset, nb_epoch=10)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
