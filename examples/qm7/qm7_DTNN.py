"""
Script that trains graph-conv models on Tox21 dataset.
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
tasks, datasets, transformers = dc.molnet.load_qm7_from_mat()
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

# Batch size of models
batch_size = 50
n_feat = [23, 23]
graph_model = dc.nn.SequentialDTNNGraph(max_n_atoms=23, n_distance=100)
graph_model.add(dc.nn.DTNNEmbedding(n_embedding=20))
graph_model.add(dc.nn.DTNNStep(n_embedding=20, n_distance=100))
graph_model.add(dc.nn.DTNNStep(n_embedding=20, n_distance=100))
graph_model.add(dc.nn.DTNNGather(n_embedding=20))

model = dc.models.MultitaskGraphRegressor(
    graph_model,
    len(tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=1e-3,
    learning_rate_decay_time=1000,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

# Fit trained model
model.fit(train_dataset, nb_epoch=10)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
