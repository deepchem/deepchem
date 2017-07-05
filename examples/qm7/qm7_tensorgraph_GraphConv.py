"""
Script that trains GraphConv models on qm7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

# Load Tox21 dataset
tasks, datasets, transformers = dc.molnet.load_qm7_from_mat(
    featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

# Batch size of models
batch_size = 64

model = dc.models.GraphConvTensorGraph(
    len(tasks), batch_size=batch_size, learning_rate=0.001, mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=50)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
