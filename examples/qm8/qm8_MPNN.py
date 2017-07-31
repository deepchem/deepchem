"""
Script that trains MPNN models on qm8 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

# Load QM8 dataset
tasks, datasets, transformers = dc.molnet.load_qm8(featurizer='MP')
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")]

# Batch size of models
batch_size = 32
n_atom_feat = 70
n_pair_feat = 8

model = dc.models.MPNNTensorGraph(
    len(tasks),
    n_atom_feat=n_atom_feat,
    n_pair_feat=n_pair_feat,
    T=5,
    M=10,
    batch_size=batch_size,
    learning_rate=0.0001,
    use_queue=False,
    mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=100)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
