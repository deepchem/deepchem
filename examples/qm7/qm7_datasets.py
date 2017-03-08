"""
qm7 dataset loader.
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


def load_qm7_from_mat(featurizer=None, split='stratified'):
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, "qm7.mat")

  if not os.path.exists(dataset_file):
    os.system('wget -P ' + current_dir +
              ' http://www.quantum-machine.org/data/qm7.mat')
  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)
  print(len(dataset))

  current_dir = os.path.dirname(os.path.realpath(__file__))
  split_file = os.path.join(current_dir, "./qm7_splits.csv")

  split_indices = []
  with open(split_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      row_int = (np.asarray(list(map(int, row)))).tolist()
      split_indices.append(row_int)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'indice': dc.splits.IndiceSplitter(valid_indices=split_indices[1]),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)
  print(len(train_dataset))
  print(len(valid_dataset))
  print(len(test_dataset))

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  qm7_tasks = np.arange(y.shape[0])
  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7b_from_mat(featurizer=None, split='stratified'):
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, "./qm7b.mat")

  if not os.path.exists(dataset_file):
    os.system('wget -P ' + current_dir +
              ' http://www.quantum-machine.org/data/qm7b.mat')
  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)

  current_dir = os.path.dirname(os.path.realpath(__file__))
  split_file = os.path.join(current_dir, "./qm7_splits.csv")

  split_indices = []
  with open(split_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      row_int = (np.asarray(list(map(int, row)))).tolist()
      split_indices.append(row_int)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'indice': dc.splits.IndiceSplitter(valid_indices=split_indices[1]),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  qm7_tasks = np.arange(y.shape[1])
  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7(featurizer=None, split='random'):
  """Load qm7 datasets."""
  # Featurize qm7 dataset
  print("About to featurize qm7 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, "./gdb7.sdf")
  qm7_tasks = ["u0_atom"]
  if featurizer is None:
    featurizer = dc.feat.CoulombMatrixEig(23)
  loader = dc.data.SDFLoader(
      tasks=qm7_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  split_file = os.path.join(current_dir, "./qm7_splits.csv")

  split_indices = []
  with open(split_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      row_int = (np.asarray(list(map(int, row)))).tolist()
      split_indices.append(row_int)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'indice': dc.splits.IndiceSplitter(valid_indices=split_indices[1]),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers
