"""
SWEET dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import logging
import deepchem as dc

logger = logging.getLogger(__name__)


def load_sweet(featurizer='ECFP', split='index', reload=True, frac_train=.8):
  """Load sweet datasets."""
  # Load Sweetlead dataset
  logger.info("About to load Sweetlead dataset.")
  data_dir = dc.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir,
                            "sweetlead/" + featurizer + "/" + str(split))

  dataset_file = os.path.join(data_dir, "sweet.csv.gz")
  if not os.path.exists(dataset_file):
    dc.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/sweet.csv.gz'
    )

  # Featurize SWEET dataset
  print("About to featurize SWEET dataset.")
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  else:
    raise ValueError("Other featurizations not supported")
  SWEET_tasks = ["task"]

  loader = dc.data.CSVLoader(
      tasks=SWEET_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return SWEET_tasks, (dataset, None, None), transformers

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
      'task': dc.splits.TaskSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  if reload:
    dc.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                       transformers)
    all_dataset = (train, valid, test)

  return SWEET_tasks, (train, valid, test), transformers
