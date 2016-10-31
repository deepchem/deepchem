"""
Train low-data attn models on random forests. Test last fold only.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tempfile
import numpy as np
import deepchem as dc
import tensorflow as tf
from datasets import load_tox21_convmol

# Number of folds for split 
K = 4
# Depth of attention module
max_depth = 4
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
# Apply a residual lstm layer
support_model.join(dc.nn.ResiLSTMEmbedding(
    test_batch_size, support_batch_size, max_depth))


with tf.Session() as sess:
  model = dc.models.SupportGraphClassifier(
    sess, support_model, test_batch_size=test_batch_size,
    support_batch_size=support_batch_size,
    learning_rate=1e-3, verbosity="high")

  ############################################################ DEBUG
  print("FIT")
  ############################################################ DEBUG
  model.fit(train_dataset, n_trials=n_train_trials,
            n_steps_per_trial=n_steps_per_trial, n_pos=n_pos,
            n_neg=n_neg, replace=False)
  model.save()

  ############################################################ DEBUG
  print("EVAL")
  ############################################################ DEBUG
  scores = model.evaluate(
      test_dataset, metric, n_pos=n_pos, n_neg=n_neg, replace=replace,
      n_trials=n_eval_trials)
  print("Scores on evaluation dataset")
  print(scores)
