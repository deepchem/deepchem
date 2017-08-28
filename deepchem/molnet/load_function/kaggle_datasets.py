"""
KAGGLE dataset loader.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time

import numpy as np
import deepchem
from deepchem.molnet.load_function.kaggle_features import merck_descriptors


def remove_missing_entries(dataset):
  """Remove missing entries.

  Some of the datasets have missing entries that sneak in as zero'd out
  feature vectors. Get rid of them.
  """
  for i, (X, y, w, ids) in enumerate(dataset.itershards()):
    available_rows = X.any(axis=1)
    print("Shard %d has %d missing entries." %
          (i, np.count_nonzero(~available_rows)))
    X = X[available_rows]
    y = y[available_rows]
    w = w[available_rows]
    ids = ids[available_rows]
    dataset.set_shard(i, X, y, w, ids)


def get_transformers(train_dataset):
  """Get transformers applied to datasets."""
  transformers = []
  #transformers = [
  #    deepchem.trans.LogTransformer(transform_X=True),
  #    deepchem.trans.NormalizationTransformer(transform_y=True,
  #                                      dataset=train_dataset)]
  return transformers


# Set shard size low to avoid memory problems.
def gen_kaggle(KAGGLE_tasks,
               train_dir,
               valid_dir,
               test_dir,
               data_dir,
               shard_size=2000):
  """Load KAGGLE datasets. Does not do train/test split"""
  ############################################################## TIMING
  time1 = time.time()
  ############################################################## TIMING
  # Set some global variables up top
  train_files = os.path.join(data_dir,
                             "KAGGLE_training_disguised_combined_full.csv.gz")
  valid_files = os.path.join(data_dir,
                             "KAGGLE_test1_disguised_combined_full.csv.gz")
  test_files = os.path.join(data_dir,
                            "KAGGLE_test2_disguised_combined_full.csv.gz")
  if not os.path.exists(train_files):
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/KAGGLE_training_disguised_combined_full.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/KAGGLE_test1_disguised_combined_full.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/KAGGLE_test2_disguised_combined_full.csv.gz'
    )

  # Featurize KAGGLE dataset
  print("About to featurize KAGGLE dataset.")
  featurizer = deepchem.feat.UserDefinedFeaturizer(merck_descriptors)

  loader = deepchem.data.UserCSVLoader(
      tasks=KAGGLE_tasks, id_field="Molecule", featurizer=featurizer)

  print("Featurizing train datasets")
  train_dataset = loader.featurize(train_files, shard_size=shard_size)

  print("Featurizing valid datasets")
  valid_dataset = loader.featurize(valid_files, shard_size=shard_size)

  print("Featurizing test datasets")
  test_dataset = loader.featurize(test_files, shard_size=shard_size)

  print("Remove missing entries from datasets.")
  remove_missing_entries(train_dataset)
  remove_missing_entries(valid_dataset)
  remove_missing_entries(test_dataset)

  print("Shuffling order of train dataset.")
  train_dataset.sparse_shuffle()

  print("Transforming datasets with transformers.")
  transformers = get_transformers(train_dataset)

  for transformer in transformers:
    print("Performing transformations with %s" % transformer.__class__.__name__)
    print("Transforming datasets")
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  print("Moving directories")
  train_dataset.move(train_dir)
  valid_dataset.move(valid_dir)
  test_dataset.move(test_dir)

  ############################################################## TIMING
  time2 = time.time()
  print("TIMING: KAGGLE fitting took %0.3f s" % (time2 - time1))
  ############################################################## TIMING

  return train_dataset, valid_dataset, test_dataset


def load_kaggle(shard_size=2000, featurizer=None, split=None, reload=True):
  """Loads kaggle datasets. Generates if not stored already."""
  KAGGLE_tasks = [
      '3A4', 'CB1', 'DPP4', 'HIVINT', 'HIV_PROT', 'LOGD', 'METAB', 'NK1', 'OX1',
      'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN'
  ]
  data_dir = deepchem.utils.get_data_dir()

  data_dir = os.path.join(data_dir, "kaggle")
  train_dir = os.path.join(data_dir, "train_dir")
  valid_dir = os.path.join(data_dir, "valid_dir")
  test_dir = os.path.join(data_dir, "test_dir")

  if (os.path.exists(train_dir) and os.path.exists(valid_dir) and
      os.path.exists(test_dir)):
    print("Reloading existing datasets")
    train_dataset = deepchem.data.DiskDataset(train_dir)
    valid_dataset = deepchem.data.DiskDataset(valid_dir)
    test_dataset = deepchem.data.DiskDataset(test_dir)
  else:
    print("Featurizing datasets")
    train_dataset, valid_dataset, test_dataset = \
      gen_kaggle(KAGGLE_tasks, train_dir, valid_dir, test_dir, data_dir,
                  shard_size=shard_size)

  transformers = get_transformers(train_dataset)
  return KAGGLE_tasks, (train_dataset, valid_dataset,
                        test_dataset), transformers
