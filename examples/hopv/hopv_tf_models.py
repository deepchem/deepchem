"""
Script that trains multitask models on HOPV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from deepchem.molnet import load_hopv

# Only for debug!
np.random.seed(123)

# Load HOPV dataset
n_features = 1024
hopv_tasks, hopv_datasets, transformers = load_hopv()
train_dataset, valid_dataset, test_dataset = hopv_datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean, mode="regression"),
    dc.metrics.Metric(
        dc.metrics.mean_absolute_error, np.mean, mode="regression")
]

model = dc.models.MultitaskRegressor(
    len(hopv_tasks),
    n_features,
    layer_sizes=[1000],
    dropouts=[.25],
    learning_rate=0.001,
    batch_size=50)

# Fit trained model
model.fit(train_dataset, nb_epoch=25)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
