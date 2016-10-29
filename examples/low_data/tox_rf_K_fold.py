"""
Train low-data models on random forests.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tempfile
import numpy as np
import deepchem as dc
from datasets import load_tox21_ecfp
from sklearn.ensemble import RandomForestClassifier
from deepchem.metrics import Metric
from deepchem.splits.task_splitter import merge_fold_datasets
from deepchem.splits.task_splitter import TaskSplitter
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.tf_keras_models.support_classifier import SupportGenerator
from deepchem.models.tf_keras_models.support_classifier import get_task_dataset_minus_support

model_dir = tempfile.mkdtemp()

# 4-fold splits
K = 4
# 10 positive/negative ligands
n_pos = 10
n_neg = 10
# 10 trials on test-set
n_trials = 10
# Sample supports without replacement (all pos/neg should be different)
replace = False

tox21_tasks, dataset, transformers = load_tox21_ecfp()

# Define metric
metric = Metric(dc.metrics.roc_auc_score, verbosity="high",
                mode="classification")

task_splitter = TaskSplitter()
fold_datasets = task_splitter.k_fold_split(dataset, K)

all_scores = {}
for fold in range(K):
  train_inds = list(set(range(K)) - set([fold]))
  train_folds = [fold_datasets[ind] for ind in train_inds]
  train_dataset = merge_fold_datasets(train_folds)
  test_dataset = fold_datasets[fold]

  fold_tasks = range(fold * len(test_dataset.get_task_names()),
                     (fold+1) * len(test_dataset.get_task_names()))

  # Get supports on test-set
  support_generator = SupportGenerator(
      test_dataset, range(len(test_dataset.get_task_names())), n_pos, n_neg,
      n_trials, replace)

  # Compute accuracies
  task_scores = {task: [] for task in range(len(test_dataset.get_task_names()))}
  for (task, support) in support_generator:
    # Train model on support
    sklearn_model = RandomForestClassifier(
        class_weight="balanced", n_estimators=50)
    model = SklearnModel(sklearn_model, model_dir)
    model.fit(support)

    # Test model
    task_dataset = get_task_dataset_minus_support(test_dataset, support, task)
    y_pred = model.predict_proba(task_dataset)
    score = metric.compute_metric(
        task_dataset.y, y_pred, task_dataset.w)
    #print("Score on task %s is %s" % (str(task), str(score)))
    task_scores[task].append(score)

  # Join information for all tasks.
  mean_task_scores = {}
  for task in range(len(test_dataset.get_task_names())):
    mean_task_scores[task] = np.mean(np.array(task_scores[task]))
  print("Fold %s" % str(fold))
  print(mean_task_scores)

  for (fold_task, task) in zip(fold_tasks, range(len(test_dataset.get_task_names()))):
    all_scores[fold_task] = mean_task_scores[task]

print("All scores")
print(all_scores)
