"""
Load datasets for Low Data processing.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import shutil
import tempfile
import numpy as np
import deepchem as dc

#sys.path.append("..")
#from muv.muv_datasets import load_muv

def load_tox21_ecfp(num_train=7200):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  verbosity = "high"
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")
  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

  loader = dc.load.DataLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer,
      verbosity=verbosity)
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  return tox21_tasks, dataset, transformers

def load_tox21_convmol(base_dir=None, num_train=7200):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  verbosity = "high"
  current_dir = os.path.dirname(os.path.realpath(__file__))
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
      featurizer=featurizer, verbosity=verbosity)
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  return tox21_tasks, dataset, transformers
