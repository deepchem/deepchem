"""
FACTORS dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import time
import numpy as np
import deepchem as dc
from factors_features import factors_descriptors 

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

def load_factors(shard_size=10000, num_shards_per_batch=4):
  """Load Factor datasets."""
  verbosity = "high"
  train_files = ("FACTORS_training_disguised_combined_full.csv.gz")
  valid_files = ("FACTORS_test1_disguised_combined_full.csv.gz")
  test_files = ("FACTORS_test2_disguised_combined_full.csv.gz")

  # Featurize FACTORS dataset
  print("About to featurize FACTORS dataset.")
  featurizer = dc.feat.UserDefinedFeaturizer(merck_descriptors)
  FACTORS_tasks = (['T_0000%d' % i for i in range(1, 10)]
                   + ['T_000%d' % i for i in range(10, 13)])

  loader = dc.load.DataLoader(
      tasks=FACTORS_tasks, id_field="Molecule",
      featurizer=featurizer, verbosity=verbosity)

  train_datasets, valid_datasets, test_datasets = [], [], []
  print("Featurizing train datasets")
  train_dataset = loader.featurize(
      train_files, 
      shard_size=shard_size, num_shards_per_batch=num_shards_per_batch)

  print("Featurizing valid datasets")
  valid_dataset = loader.featurize(
      valid_files, shard_size=shard_size)

  print("Featurizing test datasets")
  print("Creating test dataset")
  test_dataset = loader.featurize(
      test_files, shard_size=shard_size)

  print("Remove missing entries from datasets.")
  remove_missing_entries(train_dataset)
  remove_missing_entries(valid_dataset)
  remove_missing_entries(test_dataset)

  print("Transforming datasets with transformers.")
  transformers = [
      dc.trans.LogTransformer(transform_X=True),
      dc.trans.NormalizationTransformer(transform_y=True,
                                        dataset=train_dataset)]
  for transformer in transformers:
    print("Performing transformations with %s"
          % transformer.__class__.__name__)
    for dataset in [train_dataset, valid_dataset, test_dataset]:
      print("Transforming dataset")
      transformer.transform(dataset)

  print("Shuffling order of train dataset.")
  train_dataset.sparse_shuffle()
  
  return FACTORS_tasks, (train_dataset, valid_dataset, test_dataset), transformers
