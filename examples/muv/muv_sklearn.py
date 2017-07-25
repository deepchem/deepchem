"""
Script that trains Sklearn multitask models on MUV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import deepchem as dc
from muv_datasets import load_muv
from sklearn.ensemble import RandomForestClassifier

np.random.seed(123)

# Load MUV dataset
muv_tasks, muv_datasets, transformers = load_muv()
(train_dataset, valid_dataset, test_dataset) = muv_datasets

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500)
  return dc.models.SklearnModel(sklearn_model, model_dir)
model = dc.models.SingletaskToMultitask(muv_tasks, model_builder)

# Fit trained model
model.fit(train_dataset)
model.save()

# Evaluate train/test scores
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
