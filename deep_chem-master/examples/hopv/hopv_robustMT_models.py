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

n_layers = 1
n_bypass_layers = 1
nb_epoch = 25
model = dc.models.RobustMultitaskRegressor(
    len(hopv_tasks),
    train_dataset.get_data_shape()[0],
    layer_sizes=[500] * n_layers,
    bypass_layer_sizes=[50] * n_bypass_layers,
    dropouts=[.25] * n_layers,
    bypass_dropouts=[.25] * n_bypass_layers,
    weight_init_stddevs=[.02] * n_layers,
    bias_init_consts=[.5] * n_layers,
    bypass_weight_init_stddevs=[.02] * n_bypass_layers,
    bypass_bias_init_consts=[.5] * n_bypass_layers,
    learning_rate=.0003,
    weight_decay_penalty=.0001,
    weight_decay_penalty_type="l2",
    batch_size=100)

# Fit trained model
model.fit(train_dataset, nb_epoch=nb_epoch)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
