# 2017 DeepCrystal Technologies - Patrick Hop
#
# Data loading a splitting file
#
# MIT License - have fun!!
# ===========================================================

import os
import random
from collections import OrderedDict

import deepchem as dc
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)

def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

def split(dataset,
          frac_train=.80,
          frac_valid=.10,
          frac_test=.10,
          log_every_n=1000):
  """
  Splits internal compounds into train/validation/test by scaffold.
  """
  np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
  scaffolds = {}
  log("About to generate scaffolds", True)
  data_len = len(dataset)
  
  for ind, smiles in enumerate(dataset):
    if ind % log_every_n == 0:
      log("Generating scaffold %d/%d" % (ind, data_len), True)
    scaffold = generate_scaffold(smiles)
    if scaffold not in scaffolds:
      scaffolds[scaffold] = [ind]
    else:
      scaffolds[scaffold].append(ind)

  scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
  scaffold_sets = [
    scaffold_set
    for (scaffold, scaffold_set) in sorted(
        scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
  ]
  train_cutoff = frac_train * len(dataset)
  valid_cutoff = (frac_train + frac_valid) * len(dataset)
  train_inds, valid_inds, test_inds = [], [], []
  log("About to sort in scaffold sets", True)
  for scaffold_set in scaffold_sets:
    if len(train_inds) + len(scaffold_set) > train_cutoff:
      if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
        test_inds += scaffold_set
      else:
        valid_inds += scaffold_set
    else:
      train_inds += scaffold_set
  return train_inds, valid_inds, test_inds

def load_dataset(filename, whiten=False):
  f = open(filename, 'r')
  features = []
  labels = []
  tracer = 0
  for line in f:
    if tracer == 0:
      tracer += 1
      continue
    splits =  line[:-1].split(',')
    features.append(splits[-1])
    labels.append(float(splits[-2]))
  features = np.array(features)
  labels = np.array(labels, dtype='float32').reshape(-1, 1)

  train_ind, val_ind, test_ins = split(features)

  train_features = np.take(features, train_ind)
  train_labels = np.take(labels, train_ind)
  val_features = np.take(features, val_ind)
  val_labels = np.take(labels, val_ind)
  
  return train_features, train_labels, val_features, val_labels
