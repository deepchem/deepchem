"""
Script that trains Weave models on SIDER dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

sider_tasks, sider_datasets, transformers = dc.molnet.load_sider(
    featurizer='Weave')
train_dataset, valid_dataset, test_dataset = sider_datasets

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])

n_atom_feat = 75
n_pair_feat = 14
# Batch size of models
batch_size = 64
n_feat = 128
graph = dc.nn.AlternateSequentialWeaveGraph(
    batch_size,
    max_atoms=max_atoms,
    n_atom_feat=n_atom_feat,
    n_pair_feat=n_pair_feat)

graph.add(dc.nn.AlternateWeaveLayer(max_atoms, 75, 14))
#graph.add(dc.nn.AlternateWeaveLayer(max_atoms, 50, 50))
graph.add(dc.nn.Dense(n_feat, 50, activation='tanh'))
graph.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph.add(
    dc.nn.AlternateWeaveGather(
        batch_size, n_input=n_feat, gaussian_expand=True))

model = dc.models.MultitaskGraphClassifier(
    graph,
    len(sider_tasks),
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
