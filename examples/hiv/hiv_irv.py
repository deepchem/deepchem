"""
Script that trains multitask models on hiv dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from hiv_datasets import load_hiv

# Only for debug!
np.random.seed(123)

# Load hiv dataset
n_features = 512
hiv_tasks, hiv_datasets, transformers = load_hiv()
train_dataset, valid_dataset, test_dataset = hiv_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

transformer = dc.trans.IRVTransformer(10, len(hiv_tasks), train_dataset)
train_dataset = transformer.transform(train_dataset)
valid_dataset = transformer.transform(valid_dataset)

model = dc.models.TensorflowMultiTaskIRVClassifier(
    len(hiv_tasks), K=10, batch_size=50, learning_rate=0.001)

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
