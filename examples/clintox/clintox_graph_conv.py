"""
Script that trains graph-conv models on clintox dataset.
@author Caleb Geniesse
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np

from deepchem.models import GraphConvModel

np.random.seed(123)
import tensorflow as tf

tf.random.set_seed(123)
import deepchem as dc
from deepchem.molnet import load_clintox

# Load clintox dataset
clintox_tasks, clintox_datasets, transformers = load_clintox(
    featurizer='GraphConv', split='random')
train_dataset, valid_dataset, test_dataset = clintox_datasets

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# Do setup required for tf/keras models
# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 50
model = GraphConvModel(
    len(clintox_tasks), batch_size=batch_size, mode='classification')

# Fit trained model
model.fit(train_dataset, nb_epoch=10)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
