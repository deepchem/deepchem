"""
Train low-data Tox21 models with random forests. Test last fold only.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tempfile
import numpy as np
import deepchem as dc
from datasets import load_tox21_ecfp
from sklearn.ensemble import RandomForestClassifier

# 4-fold splits
K = 4
# num positive/negative ligands
n_pos = 10
n_neg = 10
# 10 trials on test-set
n_trials = 20

tox21_tasks, dataset, transformers = load_tox21_ecfp()

# Define metric
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification")

task_splitter = dc.splits.TaskSplitter()
fold_datasets = task_splitter.k_fold_split(dataset, K)

train_folds = fold_datasets[:-1] 
train_dataset = dc.splits.merge_fold_datasets(train_folds)
test_dataset = fold_datasets[-1]

# Get supports on test-set
support_generator = dc.data.SupportGenerator(
    test_dataset, n_pos, n_neg, n_trials)

# Compute accuracies
task_scores = {task: [] for task in range(len(test_dataset.get_task_names()))}
for (task, support) in support_generator:
  # Train model on support
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=100)
  model = dc.models.SklearnModel(sklearn_model)
  model.fit(support)

  # Test model
  task_dataset = dc.data.get_task_dataset_minus_support(
      test_dataset, support, task)
  y_pred = model.predict_proba(task_dataset)
  score = metric.compute_metric(
      task_dataset.y, y_pred, task_dataset.w)
  print("Score on task %s is %s" % (str(task), str(score)))
  task_scores[task].append(score)

# Join information for all tasks.
mean_task_scores = {}
std_task_scores = {}
for task in range(len(test_dataset.get_task_names())):
  mean_task_scores[task] = np.mean(np.array(task_scores[task]))
  std_task_scores[task] = np.std(np.array(task_scores[task]))

print("Mean scores")
print(mean_task_scores)
print("Standard Deviations")
print(std_task_scores)
print("Median of Mean Scores")
print(np.median(np.array(mean_task_scores.values())))
