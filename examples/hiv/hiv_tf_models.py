"""
Script that trains multitask models on hiv dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from hiv_datasets import load_hiv

# Only for debug!
np.random.seed(123)

# Load hiv dataset
n_features = 1024
hiv_tasks, hiv_datasets, transformers = load_hiv()
train_dataset, valid_dataset, test_dataset = hiv_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

model = dc.models.TensorflowMultiTaskClassifier(
    len(hiv_tasks),
    n_features,
    layer_sizes=[1000],
    dropouts=[.25],
    learning_rate=0.001,
    batch_size=50)

# Fit trained model
model.fit(train_dataset)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
