"""
BBBC Dataset loader.

This file contains image loaders for the BBBC dataset collection (https://data.broadinstitute.org/bbbc/image_sets.html).
"""

from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import logging
import deepchem

logger = logging.getLogger(__name__)


def load_bbbc001(split='index', reload=True):
  """Load BBBC001 dataset
  
  This dataset contains 6 images of human HT29 colon cancer cells. The task is to learn to predict the cell counts in these images. This dataset is too small to serve to train algorithms, but might serve as a good test dataset. https://data.broadinstitute.org/bbbc/BBBC001/
  """
  # Featurize BBBC001 dataset
  bbbc001_tasks = ["cell-count"]
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "bbbc001/" + str(split))
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return bbbc001_tasks, all_dataset, transformers
  dataset_file = os.path.join(data_dir, "BBBC001_v1_images_tif.zip")
  labels_file = os.path.join(data_dir, "BBBC001_v1_counts.txt")
  
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(
        'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_images_tif.zip'
    )
  if not os.path.exists(labels_file):
    deepchem.utils.download_url(
        'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_counts.txt'
    )
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
  counts = [(float(x[1]) + float(x[2]))/2.0 for x in lines]
  y = np.array(counts)

  # This is kludgy way to add y to dataset. Can be done better?
  dataset = deepchem.data.DiskDataset.from_numpy(dataset.X, y)

  if split == None:
    return bbbc001_tasks, (dataset, None, None), transformers

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
  return bbbc001_tasks, all_dataset, transformers
