"""
Script that trains RF model on UV datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import tempfile
import shutil
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from MERCK_datasets import load_uv

###Load data###
shard_size = 2000
num_cores = 1
num_shards_per_batch = 4
print("About to load UV data.")
UV_tasks, datasets, transformers = load_uv(
    shard_size=shard_size, num_shards_per_batch=num_shards_per_batch)
train_dataset, valid_dataset, test_dataset = datasets

print("Number of compounds in train set")
print(len(train_dataset))
print("Number of compounds in validation set")
print(len(valid_dataset))
print("Number of compounds in test set")
print(len(test_dataset))

num_features = train_dataset.get_data_shape()[0]
print("Num features: %d" % num_features)

def task_model_builder(model_dir):
  sklearn_model = RandomForestRegressor(
      n_estimators=100, max_features=int(num_features/3),
      min_samples_split=5, n_jobs=-1)
  return dc.models.SklearnModel(sklearn_model, model_dir)
model = dc.models.SingletaskToMultitask(UV_tasks, task_model_builder)

###Evaluate models###
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, task_averager=np.mean,
                           mode="regression")

print("Training model")
model.fit(train_dataset)

train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
#Only use for final evaluation
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

print("Test scores")
print(test_scores)
