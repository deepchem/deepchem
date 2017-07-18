"""
Script that trains ANI models on qm7 dataset.
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
    featurizer='BPSymmetryFunction')
train_dataset, valid_dataset, test_dataset = datasets

# Batch size of models
max_atoms = 23
batch_size = 128
layer_structures = [128, 128, 64]
atom_number_cases = [1, 6, 7, 8, 16]

ANItransformer = dc.trans.ANITransformer(
    max_atoms=max_atoms, atom_cases=atom_number_cases)
train_dataset = ANItransformer.transform(train_dataset)
valid_dataset = ANItransformer.transform(valid_dataset)
test_dataset = ANItransformer.transform(test_dataset)
n_feat = ANItransformer.get_num_feats() - 1

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

model = dc.models.ANIRegression(
    len(tasks),
    max_atoms,
    n_feat,
    layer_structures=layer_structures,
    atom_number_cases=atom_number_cases,
    batch_size=batch_size,
    learning_rate=0.001,
    use_queue=False,
    mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=300, checkpoint_interval=100)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
