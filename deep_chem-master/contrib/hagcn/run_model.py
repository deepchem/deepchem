from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from hagcn_model import HAGCN

delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
    featurizer='GraphConv', split='index')
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

max_train = max([mol.get_num_atoms() for mol in train_dataset.X])
max_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
max_test = max([mol.get_num_atoms() for mol in test_dataset.X])
max_atoms = max([max_train, max_valid, max_test])

# Args
n_atom_feat = 75
batch_size = 128
k_max = 4

model = HAGCN(
    max_nodes=max_atoms,
    n_tasks=len(delaney_tasks),
    num_node_features=n_atom_feat,
    batch_size=batch_size,
    k_max=k_max)

model.fit(dataset=train_dataset, nb_epoch=80)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
