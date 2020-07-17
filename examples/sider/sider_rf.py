"""
Script that trains Sklearn multitask models on the sider dataset
@Author Bharath Ramsundar, Aneesh Pappu
"""
import os
import shutil
import numpy as np
import deepchem as dc
#from sider_datasets import load_sider
from sklearn.ensemble import RandomForestClassifier

sider_tasks, datasets, transformers = dc.molnet.load_sider()
train_dataset, valid_dataset, test_dataset = datasets

metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean,
                           mode="classification")

def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=100)
  return dc.models.SklearnModel(sklearn_model, model_dir)
model = dc.models.SingletaskToMultitask(sider_tasks, model_builder)

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
