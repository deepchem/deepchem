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

<<<<<<< HEAD
def load_sider(method = 'ECFP'):
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
  if method == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif method == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()

=======
def load_sider():
  """Load SIDER datasets. Does not do train/test split"""
  # Featurize SIDER dataset
  print("About to featurize SIDER dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "./sider.csv.gz")
  featurizer = dc.feat.CircularFingerprint(size=1024)

  dataset = dc.utils.save.load_from_disk(dataset_file)
>>>>>>> da5be2a805bfba096fe3f029c23c2ca675a195a5
  SIDER_tasks = dataset.columns.values[1:].tolist()
  print("SIDER tasks: %s" % str(SIDER_tasks))
  print("%d tasks in total" % len(SIDER_tasks))

  loader = dc.load.DataLoader(tasks=SIDER_tasks,
                              smiles_field="smiles",
                              featurizer=featurizer,
                              verbosity='high')
  dataset = loader.featurize(dataset_file)
  print("%d datapoints in SIDER dataset" % len(dataset))

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitter = dc.splits.IndexSplitter()
  train, valid, test = splitter.train_valid_test_split(dataset,
      compute_feature_statistics=False)

  return SIDER_tasks, (train, valid, test), transformers
