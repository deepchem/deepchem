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
from deepchem.metrics import Metric
from deepchem.splits.task_splitter import merge_fold_datasets
from deepchem.splits.task_splitter import TaskSplitter
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.tf_keras_models.support_classifier import SupportGenerator
from deepchem.models.tf_keras_models.support_classifier import get_task_dataset_minus_support
from deepchem.models.tf_keras_models.keras_layers import GraphConv
from deepchem.models.tf_keras_models.keras_layers import GraphPool
from deepchem.models.tf_keras_models.keras_layers import GraphGather
from deepchem.featurizers.graph_features import ConvMolFeaturizer
from deepchem.models.tf_keras_models.graph_models import SequentialSupportGraphModel
from deepchem.models.tf_keras_models.support_classifier import SupportGraphClassifier
from keras.layers import Dense, BatchNormalization

model_dir = tempfile.mkdtemp()

# 4-fold splits
K = 4
# 10 positive/negative ligands
n_pos = 10
n_neg = 10
# Set batch sizes for network
test_batch_size = 10
support_batch_size = n_pos + n_neg
# 10 trials on test-set
n_trials = 10
# Sample supports without replacement (all pos/neg should be different)
replace = False
# Number of features on conv-mols
n_feat = 71

tox21_tasks, dataset, transformers = load_tox21_convmol()

# Define metric
metric = Metric(dc.metrics.roc_auc_score, verbosity="high",
                mode="classification")

task_splitter = TaskSplitter()
fold_datasets = task_splitter.k_fold_split(dataset, K)

train_folds = fold_datasets[:-1] 
train_dataset = merge_fold_datasets(train_folds)
test_dataset = fold_datasets[-1]

# Train support model on train
support_model = SequentialSupportGraphModel(n_feat)

# Add layers
# output will be (n_atoms, 64)
support_model.add(GraphConv(64, activation='relu'))
# Need to add batch-norm separately to test/support due to differing
# shapes.
# output will be (n_atoms, 64)
support_model.add_test(BatchNormalization(epsilon=1e-5, mode=1))
# output will be (n_atoms, 64)
support_model.add_support(BatchNormalization(epsilon=1e-5, mode=1))
support_model.add(GraphPool())
support_model.add_test(GraphGather(test_batch_size))
support_model.add_support(GraphGather(support_batch_size))

with tf.Session() as sess:
  model = SupportGraphClassifier(
    sess, support_model, len(train_dataset.get_task_names()), model_dir, 
    test_batch_size=test_batch_size, support_batch_size=support_batch_size,
    learning_rate=1e-3, learning_rate_decay_time=1000,
    optimizer_type="adam", beta1=.9, beta2=.999, verbosity="high")

  model.fit(train_dataset, nb_epoch=1, n_trials_per_epoch=10, n_pos=n_pos,
            n_neg=n_neg, replace=False)
  model.save()

  scores = model.evaluate(test_dataset, range(len(test_dataset.get_task_names())),
                          metric, n_pos=n_pos, n_neg=n_neg, replace=replace)
  print("Scores")
  print(scores)
