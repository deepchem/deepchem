"""
Script that trains multitask models on clintox dataset.
@author Caleb Geniesse
"""
import numpy as np
import deepchem as dc
from deepchem.molnet import load_clintox

# Only for debug!
np.random.seed(123)

# Load clintox dataset
n_features = 1024
clintox_tasks, clintox_datasets, transformers = load_clintox(split='random')
train_dataset, valid_dataset, test_dataset = clintox_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

model = dc.models.MultitaskClassifier(
    len(clintox_tasks),
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
