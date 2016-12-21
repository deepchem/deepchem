"""
Script that trains sklearn models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from tox21_datasets import load_tox21
from sklearn.ensemble import RandomForestClassifier

# Only for debug!
np.random.seed(123)

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21()
(train_dataset, valid_dataset, test_dataset) = tox21_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500)
  return dc.models.SklearnModel(sklearn_model, model_dir)
model = dc.models.SingletaskToMultitask(tox21_tasks, model_builder)

# Fit trained model
print("About to fit model")
model.fit(train_dataset)
model.save()

print("About to evaluate model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
