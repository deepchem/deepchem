"""
Script that trains ANI models on qm7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
import os
import tempfile

HARTREE_TO_KCAL_PER_MOL = 627.509

tasks, datasets, transformers = dc.molnet.load_qm7_from_mat(
    featurizer='BPSymmetryFunction', split='index', reload=False)
all_dataset = dc.data.DiskDataset.merge(datasets)
invalid_inds = []
X = all_dataset.X
for i in range(X.shape[0]):
  # Exclude all molecules having S
  if 16 in X[i, :, 0]:
    invalid_inds.append(i)
valid_inds = np.delete(np.arange(all_dataset.y.shape[0]), invalid_inds)
dataset = all_dataset.select(valid_inds)

splitter = dc.splits.RandomSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

y = dc.trans.undo_transforms(train.y, transformers) / HARTREE_TO_KCAL_PER_MOL
train = dc.data.DiskDataset.from_numpy(
    train.X, y, w=train.w, ids=train.ids, tasks=train.tasks)

y = dc.trans.undo_transforms(valid.y, transformers) / HARTREE_TO_KCAL_PER_MOL
valid = dc.data.DiskDataset.from_numpy(
    valid.X, y, w=valid.w, ids=valid.ids, tasks=valid.tasks)

y = dc.trans.undo_transforms(test.y, transformers) / HARTREE_TO_KCAL_PER_MOL
test = dc.data.DiskDataset.from_numpy(
    test.X, y, w=test.w, ids=test.ids, tasks=test.tasks)

# Batch size of models
max_atoms = 23
batch_size = 128
layer_structures = [64, 64, 32]
atom_number_cases = [1, 6, 7, 8]

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

model_dir = tempfile.mkdtemp()

lr_scedule = [1e-3, 1e-4, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9]

valid_best = 100.
for lr in lr_scedule:
  model = dc.models.ANIRegression(
      len(tasks),
      max_atoms,
      exp_loss=False,
      layer_structures=layer_structures,
      atom_number_cases=atom_number_cases,
      dropout=0.,
      penalty=0.,
      batch_size=batch_size,
      learning_rate=lr,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)
  if lr < 1e-3:
    model.restore()
  model.fit(train, nb_epoch=10)
  local_ct = 0
  while local_ct < 100:
    local_ct += 1
    model.fit(train, nb_epoch=1)
    train_scores = model.evaluate(train, metric)
    valid_scores = model.evaluate(valid, metric)
    print("Train MAE(kcal/mol): " +
          str(train_scores['mean_absolute_error'] * HARTREE_TO_KCAL_PER_MOL))
    print("Valid MAE(kcal/mol): " +
          str(valid_scores['mean_absolute_error'] * HARTREE_TO_KCAL_PER_MOL))
    if valid_scores['mean_absolute_error'] < valid_best:
      local_ct = 0
      valid_best = valid_scores['mean_absolute_error']
      test_scores = model.evaluate(test, metric)
      print("Test MAE(kcal/mol): " +
            str(test_scores['mean_absolute_error'] * HARTREE_TO_KCAL_PER_MOL))
