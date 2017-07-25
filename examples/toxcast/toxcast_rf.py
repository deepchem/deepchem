"""
Script that trains Sklearn multitask models on toxcast & tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from toxcast_dataset import load_toxcast
import deepchem as dc

toxcast_tasks, toxcast_datasets, transformers = load_toxcast(
    base_data_dir, reload=reload)
(train_dataset, valid_dataset, test_dataset) = toxcast_datasets

classification_metric = Metric(metrics.roc_auc_score, np.mean)

def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500, n_jobs=-1)
  return dc.models.SklearnModel(sklearn_model, model_dir)

model = SingletaskToMultitask(toxcast_tasks, model_builder)

# Fit trained model
model.fit(train_dataset)
model.save()

print("About to evaluate model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
