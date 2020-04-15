"""
Script that trains Sklearn multitask models on toxcast & tox21 dataset.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepchem.molnet import load_toxcast
import deepchem as dc

toxcast_tasks, toxcast_datasets, transformers = load_toxcast()
(train_dataset, valid_dataset, test_dataset) = toxcast_datasets

metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)


def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500, n_jobs=-1)
  return dc.models.SklearnModel(sklearn_model, model_dir)


model = dc.models.SingletaskToMultitask(toxcast_tasks, model_builder)

# Fit trained model
model.fit(train_dataset)

print("About to evaluate model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
