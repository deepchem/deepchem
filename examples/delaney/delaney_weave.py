"""
Script that trains weave models on delaney dataset.
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
    featurizer='Weave', split='index')
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])

n_atom_feat = 75
n_pair_feat = 14
# Batch size of models
batch_size = 64
n_feat = 128
graph = dc.nn.SequentialWeaveGraph(
    max_atoms=max_atoms, n_atom_feat=n_atom_feat, n_pair_feat=n_pair_feat)

graph.add(dc.nn.WeaveLayer(max_atoms, 75, 14))
#graph.add(dc.nn.WeaveLayer(max_atoms, 50, 50))
graph.add(dc.nn.WeaveConcat(batch_size, n_output=n_feat))
graph.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph.add(dc.nn.WeaveGather(batch_size, n_input=n_feat, gaussian_expand=True))

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
model.fit(train_dataset, nb_epoch=50, log_every_N_batches=50)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
