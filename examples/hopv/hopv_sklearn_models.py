"""
Script that trains sklearn models on HOPV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from hopv_datasets import load_hopv
from sklearn.ensemble import RandomForestRegressor

# Only for debug!
np.random.seed(123)

# Load HOPV dataset
hopv_tasks, hopv_datasets, transformers = load_hopv()
(train_dataset, valid_dataset, test_dataset) = hopv_datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean, mode="regression"),
    dc.metrics.Metric(
        dc.metrics.mean_absolute_error, np.mean, mode="regression")
]


def model_builder(model_dir):
  sklearn_model = RandomForestRegressor(n_estimators=500)
  return dc.models.SklearnModel(sklearn_model, model_dir)


model = dc.models.SingletaskToMultitask(hopv_tasks, model_builder)

# Fit trained model
print("About to fit model")
model.fit(train_dataset)
model.save()

print("About to evaluate model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
