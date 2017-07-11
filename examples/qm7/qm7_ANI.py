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

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

# Batch size of models
max_atoms = 23
n_hidden = 40
n_embedding = 0
batch_size = 16
atom_number_cases = [1, 6, 7, 8]

model = dc.models.ANIRegression(
    len(tasks),
    max_atoms,
    n_hidden=n_hidden,
    n_embedding=n_embedding,
    atom_number_cases=atom_number_cases,
    batch_size=batch_size,
    learning_rate=0.001,
    use_queue=False,
    mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=20, checkpoint_interval=1000)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
