"""
Script that trains multitask(torch) models on Delaney dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
sys.path.append('../../contrib/torch')
from torch_multitask_regression import TorchMultitaskRegression
import numpy as np
import deepchem as dc

# Only for debug!
np.random.seed(123)

# Load Delaney dataset
n_features = 1024
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney()
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

model = TorchMultitaskRegression(
    len(delaney_tasks),
    n_features,
    layer_sizes=[1000],
    dropouts=[.25],
    learning_rate=0.001,
    batch_size=50,
    verbosity="high")

# Fit trained model
model.fit(train_dataset, nb_epoch=10)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
