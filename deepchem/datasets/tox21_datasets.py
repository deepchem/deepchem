"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from deepchem.utils.save import load_from_disk
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.datasets import DiskDataset
from deepchem.transformers import BalancingTransformer

def load_tox21(base_dir, reload=True, num_train=7200):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  reload = True
  verbosity = "high"

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if not reload:
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

  if not reload or not os.path.exists(data_dir):
    loader = DataLoader(tasks=tox21_tasks,
                        smiles_field="smiles",
                        featurizer=featurizer,
                        verbosity=verbosity)
    dataset = loader.featurize(
        dataset_file, data_dir, shard_size=8192)
  else:
    dataset = DiskDataset(data_dir, tox21_tasks, reload=True)

  # Initialize transformers 
  transformers = [
      BalancingTransformer(transform_w=True, dataset=dataset)]
  if not reload:
    print("About to transform data")
    for transformer in transformers:
        transformer.transform(dataset)

  X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
  X_train, X_valid = X[:num_train], X[num_train:]
  y_train, y_valid = y[:num_train], y[num_train:]
  w_train, w_valid = w[:num_train], w[num_train:]
  ids_train, ids_valid = ids[:num_train], ids[num_train:]

  train_dataset = DiskDataset.from_numpy(train_dir, X_train, y_train,
                                     w_train, ids_train, tox21_tasks)
  valid_dataset = DiskDataset.from_numpy(valid_dir, X_valid, y_valid,
                                     w_valid, ids_valid, tox21_tasks)
  
  return tox21_tasks, (train_dataset, valid_dataset), transformers
