"""
Delaney dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc


def load_delaney(featurizer='ECFP', split='index'):
  """Load delaney datasets."""
  # Featurize Delaney dataset
  print("About to featurize Delaney dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir,
                              "../../datasets/delaney-processed.csv")
  delaney_tasks = ['measured log solubility in mols per litre']
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  loader = dc.data.CSVLoader(
      tasks=delaney_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Initialize transformers
  transformers = [
      dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return delaney_tasks, (train, valid, test), transformers
