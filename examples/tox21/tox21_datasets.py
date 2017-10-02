"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os

import deepchem
import numpy as np
import shutil
import deepchem as dc
from deepchem.data import DiskDataset


def load_tox21(featurizer='ECFP', split='index'):
  """Load Tox21 datasets. Does not do train/test split"""
  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))

  dataset_file = os.path.join(current_dir, "../../datasets/tox21.csv.gz")
  data_dir = deepchem.utils.get_data_dir()

  tox21_tasks = [
      'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
      'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
  ]

  dataset_dir = os.path.join(data_dir, "tox21", featurizer, split)
  train, valid, test = os.path.join(dataset_dir, 'train'), os.path.join(
      dataset_dir, 'valid'), os.path.join(dataset_dir, 'test')
  if os.path.isdir(dataset_dir):
    train, valid, test = DiskDataset(data_dir=train), DiskDataset(
        data_dir=valid), DiskDataset(data_dir=test)
    transformers = [
        dc.trans.BalancingTransformer(transform_w=True, dataset=train)
    ]
    return tox21_tasks, (train, valid, test), transformers
  if featurizer == 'ECFP':
    featurizer_func = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer_func = dc.feat.ConvMolFeaturizer()
  elif featurizer == 'AdjMatrix':
    featurizer_func = dc.feat.AdjacencyFingerprint(num_atoms_feature=True)
  loader = dc.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer_func)
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
  train, valid, test = splitter.train_valid_test_split(
      dataset, train_dir=train, valid_dir=valid, test_dir=test)

  return tox21_tasks, (train, valid, test), transformers
