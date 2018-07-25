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


def load_cell_counting(split=None, reload=True):
  """Load Cell Counting dataset.
  
  Loads the cell counting dataset from http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html.
  """
  data_dir = deepchem.utils.get_data_dir()
  # No tasks since no labels provided.
  cell_counting_tasks = []
  # For now images are loaded directly by ImageLoader
  featurizer = ""
  if reload:
    save_dir = os.path.join(data_dir,
                            "cell_counting/" + featurizer + "/" + str(split))
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return cell_counting_tasks, all_dataset, transformers
  dataset_file = os.path.join(data_dir, "cells.zip")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(
        'http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip')

  loader = deepchem.data.ImageLoader()
  dataset = loader.featurize(dataset_file)

  transformers = []

  if split == None:
    return cell_counting_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
  }
  if split not in splitters:
    raise ValueError("Only index and random splits supported.")
  splitter = splitters[split]

  train, valid, test = splitter.train_valid_test_split(dataset)
  all_dataset = (train, valid, test)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return cell_counting_tasks, all_dataset, transformers
