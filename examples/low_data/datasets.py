"""
Load datasets for Low Data processing.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import tempfile
import numpy as np
from deepchem.featurizers.graph_features import ConvMolFeaturizer
from deepchem.utils.save import load_from_disk
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.datasets import DiskDataset
from deepchem.transformers import BalancingTransformer

def load_tox21_ecfp(base_dir=None, num_train=7200):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  verbosity = "high"
  if base_dir is None:
    base_dir = tempfile.mkdtemp()

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.
  data_dir = os.path.join(base_dir, "dataset")
  train_dir = os.path.join(base_dir, "train")
  valid_dir = os.path.join(base_dir, "valid")

  # Load Tox21 dataset
  print("About to load Tox21 dataset.")
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")
  dataset = load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  featurizer = CircularFingerprint(size=1024)
  tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

  loader = DataLoader(tasks=tox21_tasks,
                      smiles_field="smiles",
                      featurizer=featurizer,
                      verbosity=verbosity)
  dataset = loader.featurize(
      dataset_file, data_dir, shard_size=8192)

  # Initialize transformers 
  transformers = [
      BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  return tox21_tasks, dataset, transformers

def load_tox21_convmol(base_dir=None, num_train=7200):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  verbosity = "high"
  if base_dir is None:
    base_dir = tempfile.mkdtemp()

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.
  data_dir = os.path.join(base_dir, "dataset")
  train_dir = os.path.join(base_dir, "train")
  valid_dir = os.path.join(base_dir, "valid")

  # Load Tox21 dataset
  print("About to load Tox21 dataset.")
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")
  dataset = load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  featurizer = ConvMolFeaturizer()
  tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

  loader = DataLoader(tasks=tox21_tasks,
                      smiles_field="smiles",
                      featurizer=featurizer,
                      verbosity=verbosity)
  dataset = loader.featurize(
      dataset_file, data_dir, shard_size=8192)

  # Initialize transformers 
  transformers = [
      BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  return tox21_tasks, dataset, transformers
