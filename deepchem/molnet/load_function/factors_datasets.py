"""
FACTOR dataset loader
"""
import os
import logging
import time

import numpy as np
import deepchem
from deepchem.molnet.load_function.kaggle_features import merck_descriptors

logger = logging.getLogger(__name__)

TRAIN_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/FACTORS_training_disguised_combined_full.csv.gz"
VALID_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/FACTORS_test1_disguised_combined_full.csv.gz"
TEST_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/FACTORS_test2_disguised_combined_full.csv.gz"

TRAIN_FILENAME = "FACTORS_training_disguised_combined_full.csv.gz"
VALID_FILENAME = "FACTORS_test1_disguised_combined_full.csv.gz"
TEST_FILENAME = "FACTORS_test2_disguised_combined_full.csv.gz"


def remove_missing_entries(dataset):
  """Remove missing entries.

  Some of the datasets have missing entries that sneak in as zero'd out
  feature vectors. Get rid of them.
  """
  for i, (X, y, w, ids) in enumerate(dataset.itershards()):
    available_rows = X.any(axis=1)
    logger.info("Shard %d has %d missing entries." %
                (i, np.count_nonzero(~available_rows)))
    X = X[available_rows]
    y = y[available_rows]
    w = w[available_rows]
    ids = ids[available_rows]
    dataset.set_shard(i, X, y, w, ids)


def get_transformers(train_dataset):
  """Gets transformers applied to the dataset"""

  transformers = list()
  # TODO: Check if anything needs to be added

  return transformers


def gen_factors(FACTORS_tasks,
                data_dir,
                train_dir,
                valid_dir,
                test_dir,
                shard_size=2000):
  """Loads the FACTORS dataset; does not do train/test split"""

  time1 = time.time()

  train_files = os.path.join(data_dir, TRAIN_FILENAME)
  valid_files = os.path.join(data_dir, VALID_FILENAME)
  test_files = os.path.join(data_dir, TEST_FILENAME)

  if not os.path.exists(train_files):
    logger.info("Downloading train file...")
    deepchem.utils.data_utils.download_url(url=TRAIN_URL, dest_dir=data_dir)
    logger.info("Training file download complete.")

    logger.info("Downloading validation file...")
    deepchem.utils.data_utils.download_url(url=VALID_URL, dest_dir=data_dir)
    logger.info("Validation file download complete.")

    logger.info("Downloading test file...")
    deepchem.utils.data_utils.download_url(url=TEST_URL, dest_dir=data_dir)
    logger.info("Test file download complete")

  # Featurize the FACTORS dataset
  logger.info("About to featurize the FACTORS dataset")
  featurizer = deepchem.feat.UserDefinedFeaturizer(merck_descriptors)
  loader = deepchem.data.UserCSVLoader(
      tasks=FACTORS_tasks, id_field="Molecule", featurizer=featurizer)

  logger.info("Featurizing the train dataset...")
  train_dataset = loader.featurize(train_files, shard_size=shard_size)

  logger.info("Featurizing the validation dataset...")
  valid_dataset = loader.featurize(valid_files, shard_size=shard_size)

  logger.info("Featurizing the test dataset...")
  test_dataset = loader.featurize(test_files, shard_size=shard_size)

  logger.info("Remove missing entries from dataset")
  remove_missing_entries(train_dataset)
  remove_missing_entries(valid_dataset)
  remove_missing_entries(test_dataset)

  # Shuffle the training data
  logger.info("Shuffling the training dataset")
  train_dataset.sparse_shuffle()

  # Apply transformations
  logger.info("Transforming datasets with transformers")
  transformers = get_transformers(train_dataset)

  for transformer in transformers:
    logger.info("Performing transformations with {}".format(
        transformer.__class__.__name__))

    logger.info("Transforming the training dataset...")
    train_dataset = transformer.transform(train_dataset)

    logger.info("Transforming the validation dataset...")
    valid_dataset = transformer.transform(valid_dataset)

    logger.info("Transforming the test dataset...")
    test_dataset = transformer.transform(test_dataset)

  logger.info("Transformations complete.")
  logger.info("Moving datasets to corresponding directories")

  train_dataset.move(train_dir)
  logger.info("Train dataset moved.")

  valid_dataset.move(valid_dir)
  logger.info("Validation dataset moved.")

  test_dataset.move(test_dir)
  logger.info("Test dataset moved.")

  time2 = time.time()

  # TIMING
  logger.info("TIMING: FACTORS fitting took %0.3f s" % (time2 - time1))

  return train_dataset, valid_dataset, test_dataset


def load_factors(shard_size=2000, featurizer=None, split=None, reload=True):
  """Loads FACTOR dataset; does not do train/test split

  The Factors dataset is an in-house dataset from Merck that was first introduced in the following paper:
  Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

  It contains 1500 Merck in-house compounds that were measured
  for IC50 of inhibition on 12 serine proteases. Unlike most of
  the other datasets featured in MoleculeNet, the Factors
  collection does not have structures for the compounds tested
  since they were proprietary Merck compounds. However, the
  collection does feature pre-computed descriptors for these
  compounds.

  Note that the original train/valid/test split from the source
  data was preserved here, so this function doesn't allow for
  alternate modes of splitting. Similarly, since the source data
  came pre-featurized, it is not possible to apply alternative
  featurizations.

  Parameters
  ----------
  shard_size: int, optional
    Size of the DiskDataset shards to write on disk
  featurizer: optional
    Ignored since featurization pre-computed
  split: optional
    Ignored since split pre-computed
  reload: bool, optional
    Whether to automatically re-load from disk

  """

  FACTORS_tasks = [
      'T_00001', 'T_00002', 'T_00003', 'T_00004', 'T_00005', 'T_00006',
      'T_00007', 'T_00008', 'T_00009', 'T_00010', 'T_00011', 'T_00012'
  ]

  data_dir = deepchem.utils.data_utils.get_data_dir()
  data_dir = os.path.join(data_dir, "factors")

  if not os.path.exists(data_dir):
    os.mkdir(data_dir)

  train_dir = os.path.join(data_dir, "train_dir")
  valid_dir = os.path.join(data_dir, "valid_dir")
  test_dir = os.path.join(data_dir, "test_dir")

  if (os.path.exists(train_dir) and os.path.exists(valid_dir) and
      os.path.exists(test_dir)):

    logger.info("Reloading existing datasets")
    train_dataset = deepchem.data.DiskDataset(train_dir)
    valid_dataset = deepchem.data.DiskDataset(valid_dir)
    test_dataset = deepchem.data.DiskDataset(test_dir)

  else:
    logger.info("Featurizing datasets")
    train_dataset, valid_dataset, test_dataset = gen_factors(
        FACTORS_tasks=FACTORS_tasks,
        data_dir=data_dir,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        shard_size=shard_size)

  transformers = get_transformers(train_dataset)

  return FACTORS_tasks, (train_dataset, valid_dataset,
                         test_dataset), transformers
