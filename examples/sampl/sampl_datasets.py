"""
SAMPL dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc

def load_sampl(featurizer='ECFP', split='index'):
  """Load SAMPL datasets."""
  # Featurize SAMPL dataset
  print("About to featurize SAMPL dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "./SAMPL.csv")
  SAMPL_tasks = ['expt']
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  loader = dc.data.CSVLoader(
      tasks=SAMPL_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter()}
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return SAMPL_tasks, (train, valid, test), transformers
