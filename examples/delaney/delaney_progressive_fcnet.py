"""
Script that trains progressive multitask models on Delaney dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import deepchem as dc
from deepchem.molnet import load_delaney

# Only for debug!
np.random.seed(123)

# Load Delaney dataset
n_features = 1024
delaney_tasks, delaney_datasets, transformers = load_delaney()
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

n_layers = 1
nb_epoch = 10
model = dc.models.ProgressiveMultitaskRegressor(
    len(delaney_tasks),
    n_features,
    layer_sizes=[1000] * n_layers,
    dropouts=[.25] * n_layers,
    alpha_init_stddevs=[.02] * n_layers,
    weight_init_stddevs=[.02] * n_layers,
    bias_init_consts=[1.] * n_layers,
    learning_rate=.001,
    batch_size=100)

# Fit trained model
model.fit(train_dataset)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
