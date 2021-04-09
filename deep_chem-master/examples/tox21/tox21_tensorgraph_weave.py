"""
Script that trains weave models on tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc

# Load tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(
    featurizer='Weave', split='index')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

n_atom_feat = 75
n_pair_feat = 14
# Batch size of models
batch_size = 64
n_feat = 128

model = dc.models.WeaveModel(
    len(tox21_tasks),
    batch_size=batch_size,
    learning_rate=1e-3,
    use_queue=False,
    mode='classification')

# Fit trained model
model.fit(train_dataset, nb_epoch=50)
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
