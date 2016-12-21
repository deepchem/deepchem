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

def load_sider(featurizer='ECFP', split='index'):
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
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()

  SIDER_tasks = dataset.columns.values[1:].tolist()
  print("SIDER tasks: %s" % str(SIDER_tasks))
  print("%d tasks in total" % len(SIDER_tasks))

  loader = dc.data.CSVLoader(
      tasks=SIDER_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  print("%d datapoints in SIDER dataset" % len(dataset))

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter()}
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  return SIDER_tasks, (train, valid, test), transformers
