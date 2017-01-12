"""
UV dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import time
import numpy as np
import deepchem as dc
from uv_features import uv_descriptors 

def remove_missing_entries(dataset):
  """Remove missing entries.

  Some of the datasets have missing entries that sneak in as zero'd out
  feature vectors. Get rid of them.
  """
  for i, (X, y, w, ids) in enumerate(dataset.itershards()):
    available_rows = X.any(axis=1)
    print("Shard %d has %d missing entries."
        % (i, np.count_nonzero(~available_rows)))
    X = X[available_rows]
    y = y[available_rows]
    w = w[available_rows]
    ids = ids[available_rows]
    dataset.set_shard(i, X, y, w, ids)

def load_uv(shard_size=10000, num_shards_per_batch=4):
  """Load UV datasets."""
  verbosity = "high"
  train_files = ("UV_training_disguised_combined_full.csv.gz")
  valid_files = ("UV_test1_disguised_combined_full.csv.gz")
  test_files = ("UV_test2_disguised_combined_full.csv.gz")

  # Featurize UV dataset
  print("About to featurize UV dataset.")
  featurizer = dc.feat.UserDefinedFeaturizer(merck_descriptors)
  UV_tasks = (['logTIC'] +
                  ['w__%d' % i for i in range(210, 401)])

  loader = dc.load.DataLoader(
      tasks=UV_tasks, id_field="Molecule",
      featurizer=featurizer, verbosity=verbosity)

  train_datasets, valid_datasets, test_datasets = [], [], []
  print("Featurizing train datasets")
  train_dataset = loader.featurize(
      train_files, shard_size=shard_size, num_shards_per_batch=num_shards_per_batch)

  print("Featurizing valid datasets")
  valid_dataset = loader.featurize(
      valid_files, shard_size=shard_size)

  print("Featurizing test datasets")
  test_dataset = loader.featurize(
      test_files, shard_size=shard_size)

  print("Remove missing entries from datasets.")
  remove_missing_entries(train_dataset)
  remove_missing_entries(valid_dataset)
  remove_missing_entries(test_dataset)

  print("Transforming datasets with transformers.")
  transformers = [
      dc.trans.LogTransformer(transform_X=True),
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)]
  for transformer in transformers:
    print("Performing transformations with %s"
          % transformer.__class__.__name__)
    for dataset in [train_dataset, valid_dataset, test_dataset]:
      print("Transforming dataset")
      transformer.transform(dataset)

  print("Shuffling order of train dataset.")
  train_dataset.sparse_shuffle()
  
  return UV_tasks, (train_dataset, valid_dataset, test_dataset), transformers
