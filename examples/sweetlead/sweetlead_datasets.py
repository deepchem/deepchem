"""
SWEET dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc

def load_sweet(base_dir, frac_train=.8):
  """Load sweet datasets. Does not do train/test split"""
  current_dir = os.path.dirname(os.path.realpath(__file__))

  # Load SWEET dataset
  dataset_file = os.path.join(
      current_dir, "./sweet.csv.gz")

  # Featurize SWEET dataset
  print("About to featurize SWEET dataset.")
  featurizer = dc.feat.CircularFingerprint(size=1024)
  SWEET_tasks = dataset.columns.values[1:].tolist()

  loader = dc.data.CSVLoader(
      tasks=SWEET_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)


  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  spliter = dc.splits.IndexSplitter()
  train, valid, test = splitter.train_valid_test_split(dataset)

  return SWEET_tasks, (train, valid, test), transformers
