"""
Script that trains multitask models on gdb7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import deepchem as dc
from gdb7_datasets import load_gdb7

# Only for debug!
np.random.seed(123)

# Load gdb7 dataset
n_features = 1024
gdb7_tasks, gdb7_datasets, transformers = load_gdb7(featurizer='ECFP', split='indice')
train_dataset, valid_dataset, test_dataset = gdb7_datasets

# Fit models
metric = [dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"), 
          dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")]

model = dc.models.TensorflowMultiTaskRegressor(
    len(gdb7_tasks), n_features, layer_sizes=[1000], dropouts=[.25],
    learning_rate=0.001, batch_size=50, verbosity="high")

# Fit trained model
model.fit(train_dataset)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
