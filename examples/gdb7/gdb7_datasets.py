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
import scipy.io
import csv

def load_gdb7_from_mat(split=0):

  if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
  dataset = scipy.io.loadmat('qm7.mat')
  
  P = dataset['P'][list(range(0,split))+list(range(split+1,5))].flatten()
  X = dataset['X'][P]
  y = dataset['T'][0,P]
  w = np.ones_like(y)
  train_dataset = dc.data.NumpyDataset(X, y, w, ids=None)
  
  Ptest = dataset['P'][split]
  X = dataset['X'][Ptest]
  y = dataset['T'][0,Ptest]
  w = np.ones_like(y)
  test_dataset = dc.data.NumpyDataset(X, y, w, ids=None)

  transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    test_dataset = transformer.transform(test_dataset)

  gdb7_tasks = ["atomization_energy"]
  return gdb7_tasks, (train_dataset, test_dataset), transformers

def load_gdb7(featurizer=None, split='random'):
  """Load gdb7 datasets."""
  # Featurize gdb7 dataset
  print("About to featurize gdb7 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "./gdb7.sdf")
  gdb7_tasks = ["u0_atom"]
  if featurizer == None:
    featurizer = dc.feat.CoulombMatrix(23)
  loader = dc.data.SDFLoader(tasks=gdb7_tasks, smiles_filed="smiles",
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
      row_int = (np.asarray(list(map(int, row)))).tolist()
      split_indices.append(row_int)
  
  
  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'indice': dc.splits.IndiceSplitter(valid_indices=split_indices[1])}
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return gdb7_tasks, (train, valid, test), transformers
