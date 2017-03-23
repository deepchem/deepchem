"""
hiv dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc


def load_hiv(featurizer='ECFP', split='index'):
  """Load hiv datasets. Does not do train/test split"""
  # Featurize hiv dataset
  print("About to featurize hiv dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, "./HIV.csv")
  hiv_tasks = ["HIV_active"]
  if featurizer == 'ECFP':
    featurizer_func = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer_func = dc.feat.ConvMolFeaturizer()
  loader = dc.data.CSVLoader(
      tasks=hiv_tasks, smiles_field="smiles", featurizer=featurizer_func)
  dataset = loader.featurize(dataset_file, shard_size=8192)
  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
      'butina': dc.splits.ButinaSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return hiv_tasks, (train, valid, test), transformers
