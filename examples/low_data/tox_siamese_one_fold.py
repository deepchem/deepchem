"""
Train low-data models on random forests. Test last fold only.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tempfile
import numpy as np
import deepchem as dc
import tensorflow as tf
from datasets import load_tox21_convmol
from deepchem.models.tf_keras_models.support_classifier import SupportGenerator
from deepchem.models.tf_keras_models.support_classifier import get_task_dataset_minus_support

model_dir = tempfile.mkdtemp()

# 12-fold splits
K = 12 
# 10 positive/negative ligands
n_pos = 10
n_neg = 10
# Set batch sizes for network
test_batch_size = 100
support_batch_size = n_pos + n_neg
n_train_trials = 2000 
n_eval_trials = 5
n_steps_per_trial = 1 
# Sample supports without replacement (all pos/neg should be different)
replace = False
# Number of features on conv-mols
n_feat = 71

tox21_tasks, dataset, transformers = load_tox21_convmol()

# Define metric
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, verbosity="high", mode="classification")

task_splitter = dc.splits.TaskSplitter()
fold_datasets = task_splitter.k_fold_split(dataset, K)

train_folds = fold_datasets[:-1] 
train_dataset = dc.splits.merge_fold_datasets(train_folds)
test_dataset = fold_datasets[-1]

# Train support model on train
support_model = dc.nn.SequentialSupportGraph(n_feat)

# Add layers
# output will be (n_atoms, 64)
support_model.add(dc.nn.GraphConv(64, activation='relu'))
# Need to add batch-norm separately to test/support due to differing
# shapes.
# output will be (n_atoms, 64)
support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
# output will be (n_atoms, 64)
support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
support_model.add(dc.nn.GraphPool())
support_model.add_test(dc.nn.GraphGather(test_batch_size))
support_model.add_support(dc.nn.GraphGather(support_batch_size))

with tf.Session() as sess:
  model = dc.models.SupportGraphClassifier(
    sess, support_model, len(test_dataset.get_task_names()), model_dir, 
    test_batch_size=test_batch_size, support_batch_size=support_batch_size,
    learning_rate=1e-3, verbosity="high")

  ############################################################ DEBUG
  print("FIT")
  ############################################################ DEBUG
  model.fit(test_dataset, n_trials=n_train_trials,
            n_steps_per_trial=n_steps_per_trial, n_pos=n_pos,
            n_neg=n_neg, replace=False)
  model.save()

  ############################################################ DEBUG
  print("EVAL")
  ############################################################ DEBUG
  scores = model.evaluate(
      test_dataset, range(len(test_dataset.get_task_names())),
      metric, n_pos=n_pos, n_neg=n_neg, replace=replace,
      n_trials=n_eval_trials)
  print("Scores on training dataset")
  print(scores)
