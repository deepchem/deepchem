"""
KINASE dataset loader
"""
import os
import logging
import time

import deepchem
from deepchem.molnet.load_function.kaggle_features import merck_descriptors
from deepchem.utils import remove_missing_entries

TRAIN_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KINASE_training_disguised_combined_full.csv.gz"
VALID_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KINASE_test1_disguised_combined_full.csv.gz"
TEST_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KINASE_test2_disguised_combined_full.csv.gz"

TRAIN_FILENAME = "KINASE_training_disguised_combined_full.csv.gz"
VALID_FILENAME = "KINASE_test1_disguised_combined_full.csv.gz"
TEST_FILENAME = "KINASE_test2_disguised_combined_full.csv.gz"

logger = logging.getLogger(__name__)


def get_transformers(train_dataset):
  """Gets transformers applied to the dataset"""
  # TODO: Check for this

  transformers = list()

  return transformers


def gen_kinase(KINASE_tasks,
               train_dir,
               valid_dir,
               test_dir,
               data_dir,
               shard_size=2000):

  time1 = time.time()

  train_files = os.path.join(data_dir, TRAIN_FILENAME)
  valid_files = os.path.join(data_dir, VALID_FILENAME)
  test_files = os.path.join(data_dir, TEST_FILENAME)

  # Download files if they don't exist

  if not os.path.exists(train_files):

    logger.info("Downloading training file...")
    deepchem.utils.data_utils.download_url(url=TRAIN_URL, dest_dir=data_dir)
    logger.info("Training file download complete.")

    logger.info("Downloading validation file...")
    deepchem.utils.data_utils.download_url(url=VALID_URL, dest_dir=data_dir)
    logger.info("Validation file download complete.")

    logger.info("Downloading test file...")
    deepchem.utils.data_utils.download_url(url=TEST_URL, dest_dir=data_dir)
    logger.info("Test file download complete")

  # Featurize the KINASE dataset
  logger.info("About to featurize KINASE dataset.")
  featurizer = deepchem.feat.UserDefinedFeaturizer(merck_descriptors)

  loader = deepchem.data.UserCSVLoader(
      tasks=KINASE_tasks, id_field="Molecule", featurizer=featurizer)

  logger.info("Featurizing train datasets...")
  train_dataset = loader.featurize(
      input_files=train_files, shard_size=shard_size)

  logger.info("Featurizing validation datasets...")
  valid_dataset = loader.featurize(
      input_files=valid_files, shard_size=shard_size)

  logger.info("Featurizing test datasets....")
  test_dataset = loader.featurize(input_files=test_files, shard_size=shard_size)

  logger.info("Remove missing entries from dataset")
  remove_missing_entries(train_dataset)
  remove_missing_entries(valid_dataset)
  remove_missing_entries(test_dataset)

  # Shuffle the training data
  logger.info("Shuffling the training dataset")
  train_dataset.sparse_shuffle()

  # Apply transformations
  logger.info("Transformating datasets with transformers")
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

  logger.info("TIMING: KINASE fitting took %0.3f s" % (time2 - time1))

  return train_dataset, valid_dataset, test_dataset


def load_kinase(shard_size=2000, featurizer=None, split=None, reload=True):
  """Loads Kinase datasets, does not do train/test split

  The Kinase dataset is an in-house dataset from Merck that was first introduced in the following paper:
  Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

  It contains 2500 Merck in-house compounds that were measured
  for IC50 of inhibition on 99 protein kinases. Unlike most of
  the other datasets featured in MoleculeNet, the Kinase
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

  KINASE_tasks = [
      'T_00013', 'T_00014', 'T_00015', 'T_00016', 'T_00017', 'T_00018',
      'T_00019', 'T_00020', 'T_00021', 'T_00022', 'T_00023', 'T_00024',
      'T_00025', 'T_00026', 'T_00027', 'T_00028', 'T_00029', 'T_00030',
      'T_00031', 'T_00032', 'T_00033', 'T_00034', 'T_00035', 'T_00036',
      'T_00037', 'T_00038', 'T_00039', 'T_00040', 'T_00041', 'T_00042',
      'T_00043', 'T_00044', 'T_00045', 'T_00046', 'T_00047', 'T_00048',
      'T_00049', 'T_00050', 'T_00051', 'T_00052', 'T_00053', 'T_00054',
      'T_00055', 'T_00056', 'T_00057', 'T_00058', 'T_00059', 'T_00060',
      'T_00061', 'T_00062', 'T_00063', 'T_00064', 'T_00065', 'T_00066',
      'T_00067', 'T_00068', 'T_00069', 'T_00070', 'T_00071', 'T_00072',
      'T_00073', 'T_00074', 'T_00075', 'T_00076', 'T_00077', 'T_00078',
      'T_00079', 'T_00080', 'T_00081', 'T_00082', 'T_00083', 'T_00084',
      'T_00085', 'T_00086', 'T_00087', 'T_00088', 'T_00089', 'T_00090',
      'T_00091', 'T_00092', 'T_00093', 'T_00094', 'T_00095', 'T_00096',
      'T_00097', 'T_00098', 'T_00099', 'T_00100', 'T_00101', 'T_00102',
      'T_00103', 'T_00104', 'T_00105', 'T_00106', 'T_00107', 'T_00108',
      'T_00109', 'T_00110', 'T_00111'
  ]

  data_dir = deepchem.utils.data_utils.get_data_dir()
  data_dir = os.path.join(data_dir, "kinase")

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
    train_dataset, valid_dataset, test_dataset = gen_kinase(
        KINASE_tasks=KINASE_tasks,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        data_dir=data_dir,
        shard_size=shard_size)

  transformers = get_transformers(train_dataset)

  return KINASE_tasks, (train_dataset, valid_dataset,
                        test_dataset), transformers
