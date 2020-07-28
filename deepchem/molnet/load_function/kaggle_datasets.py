"""
KAGGLE dataset loader.
"""
import os
import logging
import time

import numpy as np
import deepchem
from deepchem.molnet.load_function.kaggle_features import merck_descriptors

logger = logging.getLogger(__name__)


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
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KAGGLE_training_disguised_combined_full.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KAGGLE_test1_disguised_combined_full.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KAGGLE_test2_disguised_combined_full.csv.gz",
        dest_dir=data_dir)

  # Featurize KAGGLE dataset
  logger.info("About to featurize KAGGLE dataset.")
  featurizer = deepchem.feat.UserDefinedFeaturizer(merck_descriptors)

  loader = deepchem.data.UserCSVLoader(
      tasks=KAGGLE_tasks, id_field="Molecule", featurizer=featurizer)

  logger.info("Featurizing train datasets")
  train_dataset = loader.featurize(train_files, shard_size=shard_size)

  logger.info("Featurizing valid datasets")
  valid_dataset = loader.featurize(valid_files, shard_size=shard_size)

  logger.info("Featurizing test datasets")
  test_dataset = loader.featurize(test_files, shard_size=shard_size)

  logger.info("Remove missing entries from datasets.")
  remove_missing_entries(train_dataset)
  remove_missing_entries(valid_dataset)
  remove_missing_entries(test_dataset)

  logger.info("Shuffling order of train dataset.")
  train_dataset.sparse_shuffle()

  logger.info("Transforming datasets with transformers.")
  transformers = get_transformers(train_dataset)

  for transformer in transformers:
    logger.info(
        "Performing transformations with %s" % transformer.__class__.__name__)
    logger.info("Transforming datasets")
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  logger.info("Moving directories")
  train_dataset.move(train_dir)
  valid_dataset.move(valid_dir)
  test_dataset.move(test_dir)

  ############################################################## TIMING
  time2 = time.time()
  logger.info("TIMING: KAGGLE fitting took %0.3f s" % (time2 - time1))
  ############################################################## TIMING

  return train_dataset, valid_dataset, test_dataset


def load_kaggle(shard_size=2000, featurizer=None, split=None, reload=True):
  """Loads kaggle datasets. Generates if not stored already.

  The Kaggle dataset is an in-house dataset from Merck that was first introduced in the following paper:

  Ma, Junshui, et al. "Deep neural nets as a method for quantitative structure–activity relationships." Journal of chemical information and modeling 55.2 (2015): 263-274.

  It contains 100,000 unique Merck in-house compounds that were
  measured on 15 enzyme inhibition and ADME/TOX datasets.
  Unlike most of the other datasets featured in MoleculeNet,
  the Kaggle collection does not have structures for the
  compounds tested since they were proprietary Merck compounds.
  However, the collection does feature pre-computed descriptors
  for these compounds.

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
  KAGGLE_tasks = [
      '3A4', 'CB1', 'DPP4', 'HIVINT', 'HIV_PROT', 'LOGD', 'METAB', 'NK1', 'OX1',
      'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN'
  ]
  data_dir = deepchem.utils.get_data_dir()

  data_dir = os.path.join(data_dir, "kaggle")
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
    train_dataset, valid_dataset, test_dataset = \
      gen_kaggle(KAGGLE_tasks, train_dir, valid_dir, test_dir, data_dir,
                  shard_size=shard_size)

  transformers = get_transformers(train_dataset)
  return KAGGLE_tasks, (train_dataset, valid_dataset,
                        test_dataset), transformers
