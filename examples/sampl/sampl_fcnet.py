"""
Script that trains multitask models on SAMPL dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import deepchem as dc
from deepchem.molnet import load_sampl

# Only for debug!
np.random.seed(123)

# Load SAMPL dataset
n_features = 1024
SAMPL_tasks, SAMPL_datasets, transformers = load_sampl()
train_dataset, valid_dataset, test_dataset = SAMPL_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

model = dc.models.MultitaskRegressor(
    len(SAMPL_tasks),
    n_features,
    layer_sizes=[1000],
    dropouts=[.25],
    learning_rate=0.001,
    batch_size=50)

# Fit trained model
model.fit(train_dataset)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
