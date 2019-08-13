"""
Cell Counting Dataset.

Loads the cell counting dataset from
http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html. Labels aren't
available for this dataset, so only raw images are provided.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
DATASET_URL = 'http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip'


def load_cell_counting(split=None,
                       reload=True,
                       data_dir=None,
                       save_dir=None,
                       **kwargs):
  """Load Cell Counting dataset.

  Loads the cell counting dataset from http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html.
  """
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR
  # No tasks since no labels provided.
  cell_counting_tasks = []
  # For now images are loaded directly by ImageLoader
  featurizer = ""
  if reload:
    save_folder = os.path.join(save_dir,
                               "cell_counting-featurized/" + str(split))
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return cell_counting_tasks, all_dataset, transformers
  dataset_file = os.path.join(data_dir, "cells.zip")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=DATASET_URL, dest_dir=data_dir)

  loader = deepchem.data.ImageLoader()
  dataset = loader.featurize(dataset_file)

  transformers = []

  if split == None:
    logger.info("Split is None, no transformers used.")
    return cell_counting_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
  }
  if split not in splitters:
    raise ValueError("Only index and random splits supported.")
  splitter = splitters[split]

  logger.info("About to split dataset with {} splitter.".format(split))
  train, valid, test = splitter.train_valid_test_split(dataset)
  transformers = []
  all_dataset = (train, valid, test)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return cell_counting_tasks, all_dataset, transformers
