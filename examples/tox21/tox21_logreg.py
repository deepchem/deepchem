"""
Script that trains multitask models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import deepchem as dc
from deepchem.molnet import load_tox21
from sklearn.linear_model import LogisticRegression

# Only for debug!
np.random.seed(123)

# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)


def model_builder(model_dir_logreg):
  sklearn_model = LogisticRegression(
      penalty="l2", C=1. / 0.05, class_weight="balanced", n_jobs=-1)
  return dc.models.sklearn_models.SklearnModel(sklearn_model, model_dir_logreg)


model = dc.models.multitask.SingletaskToMultitask(tox21_tasks, model_builder)

# Fit trained model
model.fit(train_dataset)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
