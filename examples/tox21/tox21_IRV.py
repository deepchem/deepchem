"""
Script that trains multitask models on Tox21 dataset.
"""
import os
import shutil
import numpy as np
import deepchem as dc
import time
from deepchem.molnet import load_tox21

# Only for debug!
np.random.seed(123)

# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets
K = 10
# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
transformers = [dc.trans.IRVTransformer(K, len(tox21_tasks), train_dataset)]

for transformer in transformers:
  train_dataset = transformer.transform(train_dataset)
  valid_dataset = transformer.transform(valid_dataset)
  test_dataset = transformer.transform(test_dataset)

model = dc.models.TensorflowMultitaskIRVClassifier(
    len(tox21_tasks), K=K, learning_rate=0.001, penalty=0.05, batch_size=32)

# Fit trained model
model.fit(train_dataset, nb_epoch=10)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
