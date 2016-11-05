"""
SWEET dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from deepchem.utils.save import load_from_disk
from deepchem.datasets import DiskDataset
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import BalancingTransformer

def load_sweet(base_dir, reload=True, frac_train=.8):
  """Load sweet datasets. Does not do train/test split"""
  # Set some global variables up top
  reload = True
  verbosity = "high"
  model = "logistic"
  regen = False

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
  train_dir = os.path.join(base_dir, "train_dataset")
  valid_dir = os.path.join(base_dir, "valid_dataset")

  # Load SWEET dataset
  print("About to load SWEET dataset.")
  dataset_file = os.path.join(
      current_dir, "./sweet.csv.gz")
  dataset = load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize SWEET dataset
  print("About to featurize SWEET dataset.")
  featurizer = CircularFingerprint(size=1024)
  SWEET_tasks = dataset.columns.values[1:].tolist()

  loader = DataLoader(tasks=SWEET_tasks,
                      smiles_field="smiles",
                      featurizer=featurizer,
                      verbosity=verbosity)
  if not reload or not os.path.exists(data_dir):
    dataset = loader.featurize(dataset_file, data_dir)
    regen = True
  else:
    dataset = DiskDataset(data_dir, reload=True)

  # Initialize transformers 
  transformers = [
      BalancingTransformer(transform_w=True, dataset=dataset)]
  if regen:
    print("About to transform data")
    for transformer in transformers:
        dataset = transformer.transform(dataset)

  X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
  num_tasks = 17
  num_train = frac_train * len(dataset)
  SWEET_tasks = SWEET_tasks[:num_tasks]
  print("Using following tasks")
  print(SWEET_tasks)
  X_train, X_valid = X[:num_train], X[num_train:]
  y_train, y_valid = y[:num_train, :num_tasks], y[num_train:, :num_tasks]
  w_train, w_valid = w[:num_train, :num_tasks], w[num_train:, :num_tasks]
  ids_train, ids_valid = ids[:num_train], ids[num_train:]

  train_dataset = DiskDataset.from_numpy(train_dir, X_train, y_train,
                                     w_train, ids_train, SWEET_tasks)
  valid_dataset = DiskDataset.from_numpy(valid_dir, X_valid, y_valid,
                                     w_valid, ids_valid, SWEET_tasks)
  
  return SWEET_tasks, (train_dataset, valid_dataset), transformers
