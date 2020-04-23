"""
Script that trains multitask models on Delaney dataset.
"""
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

model = dc.models.MultitaskRegressor(
    len(delaney_tasks),
    n_features,
    layer_sizes=[1000],
    dropouts=[.25],
    learning_rate=0.001,
    batch_size=50,
    verbosity="high")

# Fit trained model
model.fit(train_dataset)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
