"""
gdb7 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc
import csv

def load_gdb7(featurizer=None, split='random'):
  """Load gdb7 datasets."""
  # Featurize gdb7 dataset
  print("About to featurize gdb7 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "./gdb7.sdf")
  gdb7_tasks = ["u0_atom"]
  if featurizer is None:
    featurizer = dc.feat.CoulombMatrixEig(23)
  loader = dc.data.SDFLoader(tasks=gdb7_tasks, smiles_field="smiles", 
                             mol_field="mol", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
 
  # Initialize transformers 
  transformers = [
      dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)
  
  split_file = os.path.join(
      current_dir, "./gdb7_splits.csv")

  split_indices = []
  with open(split_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      row_int = (np.asarray(list(map(int, row)))-1).tolist()
      split_indices.append(row_int)
  
  
  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'indice': dc.splits.IndiceSplitter(valid_indices=split_indices[1])}
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return gdb7_tasks, (train, valid, test), transformers
