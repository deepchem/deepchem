"""
SIDER dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc

'''
from deepchem.utils.save import load_from_disk
from deepchem.datasets import DiskDataset
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import BalancingTransformer
'''

def load_sider():
  current_dir = os.path.dirname(os.path.realpath(__file__))

  # Load SIDER dataset
  print("About to load SIDER dataset.")
  dataset_file = os.path.join(
      current_dir, "./sider.csv.gz")
  dataset = dc.utils.save.load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize SIDER dataset
  print("About to featurize SIDER dataset.")
  featurizer = dc.feat.CircularFingerprint(size=1024)
  SIDER_tasks = dataset.columns.values[1:].tolist()

  loader = dc.load.DataLoader(tasks=SIDER_tasks,
                              smiles_field="smiles",
                              featurizer=featurizer,
                              verbosity='high')
  dataset = loader.featurize(dataset_file)

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  splitter = dc.splits.IndexSplitter()
  train, valid, test = splitter.train_valid_test_split(dataset)

  return SIDER_tasks, (train, valid, test), transformers
