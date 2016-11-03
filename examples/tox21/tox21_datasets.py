"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc

def load_tox21():
  """Load Tox21 datasets. Does not do train/test split"""
  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")
  tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

  loader = dc.load.DataLoader(
      tasks=tox21_tasks, smiles_field="smiles",
      featurizer=dc.feat.CircularFingerprint(size=1024))
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  splitter = dc.splits.IndexSplitter()
  train, valid, test = splitter.train_valid_test_split(dataset)
  return tox21_tasks, (train, valid, test), transformers

def load_tox21_convmol():
  """Load Tox21 datasets with conv feat. Does not do train/test split"""
  # Create some directories for analysis
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.

  # Load Tox21 dataset
  print("About to load Tox21 dataset.")
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")

  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  featurizer = dc.feat.ConvMolFeaturizer()
  tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

  loader = dc.load.DataLoader(
      tasks=tox21_tasks, smiles_field="smiles",
      featurizer=featurizer)
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  splitter = dc.splits.IndexSplitter()
  train, valid, test = splitter.train_valid_test_split(
      dataset, compute_feature_statistics=False)

  return tox21_tasks, (train, valid, test), transformers
