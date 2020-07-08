"""
BBBC Dataset loader.

This file contains image loaders for the BBBC dataset collection (https://data.broadinstitute.org/bbbc/image_sets.html).
"""
import os
import numpy as np
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
BBBC1_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_images_tif.zip'
BBBC1_LABEL_URL = 'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_counts.txt'

BBBC2_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_images.zip'
BBBC2_LABEL_URL = 'https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_counts.txt'


def load_bbbc001(split='index',
                 reload=True,
                 data_dir=None,
                 save_dir=None,
                 **kwargs):
  """Load BBBC001 dataset

  This dataset contains 6 images of human HT29 colon cancer cells. The task is
  to learn to predict the cell counts in these images. This dataset is too small
  to serve to train algorithms, but might serve as a good test dataset.
  https://data.broadinstitute.org/bbbc/BBBC001/
  """
  # Featurize BBBC001 dataset
  bbbc001_tasks = ["cell-count"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "bbbc001-featurized", str(split))
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return bbbc001_tasks, all_dataset, transformers
  dataset_file = os.path.join(data_dir, "BBBC001_v1_images_tif.zip")
  labels_file = os.path.join(data_dir, "BBBC001_v1_counts.txt")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=BBBC1_IMAGE_URL, dest_dir=data_dir)
  if not os.path.exists(labels_file):
    deepchem.utils.download_url(url=BBBC1_LABEL_URL, dest_dir=data_dir)
  # Featurize Images into NumpyArrays
  loader = deepchem.data.ImageLoader()
  dataset = loader.featurize(dataset_file, in_memory=False)

  # Load text file with labels
  with open(labels_file) as f:
    content = f.readlines()
  # Strip the first line which holds field labels
  lines = [x.strip() for x in content][1:]
  # Format is: Image_name count1 count2
  lines = [x.split("\t") for x in lines]
  counts = [(float(x[1]) + float(x[2])) / 2.0 for x in lines]
  y = np.array(counts)

  # This is kludgy way to add y to dataset. Can be done better?
  dataset = deepchem.data.DiskDataset.from_numpy(dataset.X, y)

  if split == None:
    transformers = []
    logger.info("Split is None, no transformers used for the dataset.")
    return bbbc001_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
  }
  if split not in splitters:
    raise ValueError("Only index and random splits supported.")
  splitter = splitters[split]

  logger.info("About to split dataset with {} splitter.".format(split))
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)
  transformers = []
  all_dataset = (train, valid, test)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return bbbc001_tasks, all_dataset, transformers


def load_bbbc002(split='index',
                 reload=True,
                 data_dir=None,
                 save_dir=None,
                 **kwargs):
  """Load BBBC002 dataset

  This dataset contains data corresponding to 5 samples of Drosophilia Kc167
  cells. There are 10 fields of view for each sample, each an image of size
  512x512. Ground truth labels contain cell counts for this dataset. Full
  details about this dataset are present at
  https://data.broadinstitute.org/bbbc/BBBC002/.
  """
  # Featurize BBBC002 dataset
  bbbc002_tasks = ["cell-count"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "bbbc002-featurized", str(split))
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return bbbc002_tasks, all_dataset, transformers
  dataset_file = os.path.join(data_dir, "BBBC002_v1_images.zip")
  labels_file = os.path.join(data_dir, "BBBC002_v1_counts.txt")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=BBBC2_IMAGE_URL, dest_dir=data_dir)
  if not os.path.exists(labels_file):
    deepchem.utils.download_url(url=BBBC2_LABEL_URL, dest_dir=data_dir)
  # Featurize Images into NumpyArrays
  loader = deepchem.data.ImageLoader()
  dataset = loader.featurize(dataset_file, in_memory=False)

  # Load text file with labels
  with open(labels_file) as f:
    content = f.readlines()
  # Strip the first line which holds field labels
  lines = [x.strip() for x in content][1:]
  # Format is: Image_name count1 count2
  lines = [x.split("\t") for x in lines]
  counts = [(float(x[1]) + float(x[2])) / 2.0 for x in lines]
  y = np.reshape(np.array(counts), (len(counts), 1))
  ids = [x[0] for x in lines]

  # This is kludgy way to add y to dataset. Can be done better?
  dataset = deepchem.data.DiskDataset.from_numpy(dataset.X, y, ids=ids)

  if split == None:
    transformers = []
    logger.info("Split is None, no transformers used for the dataset.")
    return bbbc002_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
  }
  if split not in splitters:
    raise ValueError("Only index and random splits supported.")
  splitter = splitters[split]

  logger.info("About to split dataset with {} splitter.".format(split))
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)
  all_dataset = (train, valid, test)
  transformers = []
  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return bbbc002_tasks, all_dataset, transformers
