"""
Script that trains DAG models on Tox21 dataset.
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
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(
    featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])

# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 32
graph = dc.nn.SequentialDAGGraph(75, batch_size=batch_size, max_atoms=max_atoms)
graph.add(dc.nn.DAGLayer(30, 75, max_atoms=max_atoms))
graph.add(dc.nn.DAGGather(max_atoms=max_atoms))

model = dc.models.MultitaskGraphClassifier(
    graph,
    len(tox21_tasks),
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
