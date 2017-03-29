"""
Script that trains DAG models on delaney dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

# Load Delaney dataset
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
    featurizer='GraphConv', split='random')
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Do setup required for tf/keras models
# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 32
graph = dc.nn.SequentialDAGGraph(75, batch_size=batch_size, max_atoms=55)
graph.add(dc.nn.DAGLayer(30, 75, max_atoms=55))
graph.add(dc.nn.DAGGather(max_atoms=55))

model = dc.models.MultitaskGraphRegressor(
    graph,
    len(delaney_tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=1e-3,
    learning_rate_decay_time=1000,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

# Fit trained model
model.fit(train_dataset, nb_epoch=20, log_every_N_batches=5)
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
