"""
Script that trains Keras multitask models on MUV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc
from muv_datasets import load_muv

# Set some global variables up top
np.random.seed(123)

# Load MUV data
muv_tasks, muv_datasets, transformers = load_muv()
train_dataset, valid_dataset, test_dataset = muv_datasets 
n_features = 1024 


# Build model
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

keras_model = dc.models.MultiTaskDNN(
    len(muv_tasks), n_features, "classification",
    dropout=.25, learning_rate=.001, decay=1e-4)
model = dc.models.KerasModel(keras_model, verbosity="high")

# Fit trained model
model.fit(train_dataset)
model.save()

train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
