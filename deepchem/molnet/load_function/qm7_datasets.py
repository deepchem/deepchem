"""
qm7 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import deepchem
import scipy.io


def load_qm7_from_mat(featurizer='CoulombMatrix',
                      split='stratified',
                      reload=True):
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "qm7/" + featurizer + "/" + split)

  qm7_tasks = ["u0_atom"]

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return qm7_tasks, all_dataset, transformers

  if featurizer == 'CoulombMatrix':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.mat'
      )
    dataset = scipy.io.loadmat(dataset_file)
    X = dataset['X']
    y = dataset['T']
    w = np.ones_like(y)
    dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)
  elif featurizer == 'BPSymmetryFunction':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.mat'
      )
    dataset = scipy.io.loadmat(dataset_file)
    X = np.concatenate([np.expand_dims(dataset['Z'], 2), dataset['R']], axis=2)
    y = dataset['T']
    w = np.ones_like(y)
    dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)
  else:
    dataset_file = os.path.join(data_dir, "qm7.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.csv'
      )
    if featurizer == 'ECFP':
      featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
      featurizer = deepchem.feat.RawFeaturizer()
    loader = deepchem.data.CSVLoader(
        tasks=qm7_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
  }

  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(
        save_dir, train_dataset, valid_dataset, test_dataset, transformers)

  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7b_from_mat(featurizer='CoulombMatrix',
                       split='stratified',
                       reload=True):
  data_dir = deepchem.utils.get_data_dir()
  dataset_file = os.path.join(data_dir, "qm7b.mat")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat'
    )
  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  qm7_tasks = np.arange(y.shape[1])
  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7(featurizer='CoulombMatrix', split='random', reload=True):
  """Load qm7 datasets."""
  # Featurize qm7 dataset
  print("About to featurize qm7 dataset.")
  data_dir = deepchem.utils.get_data_dir()
  dataset_file = os.path.join(data_dir, "gdb7.sdf")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb7.tar.gz'
    )
    deepchem.utils.untargz_file(os.path.join(data_dir, 'gdb7.tar.gz'), data_dir)

  qm7_tasks = ["u0_atom"]
  if featurizer == 'CoulombMatrix':
    featurizer = deepchem.feat.CoulombMatrixEig(23)
  loader = deepchem.data.SDFLoader(
      tasks=qm7_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers
