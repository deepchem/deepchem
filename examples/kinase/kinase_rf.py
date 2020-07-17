"""
Script that trains RF model on KINASE datasets.
"""
import os
import numpy as np
import tempfile
import shutil
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor

###Load data###
np.random.seed(123)
shard_size = 2000
num_trials = 1
print("About to load KINASE data.")
KINASE_tasks, datasets, transformers = dc.molnet.load_kinase(shard_size=shard_size)
train_dataset, valid_dataset, test_dataset = datasets

print("Number of compounds in train set")
print(len(train_dataset))
print("Number of compounds in validation set")
print(len(valid_dataset))
print("Number of compounds in test set")
print(len(test_dataset))

num_features = train_dataset.get_data_shape()[0]
print("Num features: %d" % num_features)

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, task_averager=np.mean)

def task_model_builder(model_dir):
  sklearn_model = RandomForestRegressor(
      n_estimators=100, max_features=int(num_features/3),
      min_samples_split=5, n_jobs=-1)
  return dc.models.SklearnModel(sklearn_model, model_dir)

all_results = []
for trial in range(num_trials):
  print("Starting trial %d" % trial)
  model = dc.models.SingletaskToMultitask(KINASE_tasks, task_model_builder)

  print("Training model")
  model.fit(train_dataset)

  print("Evaluating models")
  train_score, train_task_scores = model.evaluate(
      train_dataset, [metric], transformers, per_task_metrics=True)
  valid_score, valid_task_scores = model.evaluate(
      valid_dataset, [metric], transformers, per_task_metrics=True)
  test_score, test_task_scores = model.evaluate(
      test_dataset, [metric], transformers, per_task_metrics=True)

  all_results.append((train_score, train_task_scores,
                      valid_score, valid_task_scores,
                      test_score, test_task_scores))

  print("-----------------------------------------------------")
  print("Scores for trial %d" % trial)
  print("-----------------------------------------------------")
  print("train_task_scores")
  print(train_task_scores)
  print("Mean Train score")
  print(train_score)
  print("valid_task_scores")
  print(valid_task_scores)
  print("Mean Validation score")
  print(valid_score)
  print("test_task_scores")
  print(test_task_scores)
  print("Mean Test score")
  print(test_score)

print("#######################################################")

for trial in range(num_trials):
  (train_score, train_task_scores, valid_score, valid_task_scores,
   test_score, test_task_scores) = all_results[trial]
  print("-----------------------------------------------------")
  print("Scores for trial %d" % trial)
  print("-----------------------------------------------------")
  print("train_task_scores")
  print(train_task_scores)
  print("Mean Train score")
  print(train_score)
  print("valid_task_scores")
  print(valid_task_scores)
  print("Mean Validation score")
  print(valid_score)
  print("test_task_scores")
  print(test_task_scores)
  print("Mean Test score")
  print(test_score)
