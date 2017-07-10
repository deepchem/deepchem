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
    featurizer='BPSymmetryFunction')
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

# Batch size of models
batch_size = 64
n_embedding = 0
max_atoms = 23
n_hidden = 3
lambd = -1.
ita_radial = 0.2
ita_angular = 0.2
zeta = 0.8
n_feat = n_hidden
graph_model = dc.nn.BPSymmetryFunctionGraph(max_atoms)
graph_model.add(dc.nn.DistanceMatrix(max_atoms))
graph_model.add(dc.nn.DistanceCutoff(max_atoms, cutoff=100.))
graph_model.add(dc.nn.RadialSymmetry(max_atoms, ita=ita_radial))
graph_model.add(
    dc.nn.AngularSymmetry(max_atoms, lambd=lambd, ita=ita_angular, zeta=zeta))
graph_model.add(dc.nn.DTNNEmbedding(n_embedding=n_embedding))
graph_model.add(dc.nn.BPFeatureMerge(max_atoms))
graph_model.add(dc.nn.BatchNormalization(mode=1))
graph_model.add(dc.nn.Dense(n_hidden, n_embedding + 2))
graph_model.add(dc.nn.Dense(len(tasks), n_hidden))
graph_model.add(dc.nn.BPGather(max_atoms))

model = dc.models.DTNNMultitaskGraphRegressor(
    graph_model,
    len(tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=0.1,
    learning_rate_decay_time=1000,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

# Fit trained model
model.fit(train_dataset, nb_epoch=10)
valid_scores = model.evaluate(valid_dataset, metric, transformers)
print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
