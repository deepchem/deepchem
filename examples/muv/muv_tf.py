"""
Script that trains TF multitask models on MUV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc
from muv_datasets import load_muv

np.random.seed(123)

# Load MUV data
muv_tasks, muv_datasets, transformers = load_muv()
train_dataset, valid_dataset, test_dataset = muv_datasets 

# Build model
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean,
                           mode="classification")

model = dc.models.TensorflowMultiTaskClassifier(
    len(muv_tasks), n_features=1024, dropouts=[.25],
    learning_rate=0.001, weight_init_stddevs=[.1],
    batch_size=64, verbosity="high")

# Fit trained model
model.fit(train_dataset)

# Evaluate train/test scores
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
