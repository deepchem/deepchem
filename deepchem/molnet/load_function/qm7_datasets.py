"""
qm7 dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import deepchem
import scipy.io
import logging

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
QM7_MAT_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.mat'
QM7_CSV_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.csv'
QM7B_MAT_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat'
GDB7_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb7.tar.gz'


def load_qm7_from_mat(featurizer='CoulombMatrix',
                      split='stratified',
                      reload=True,
                      move_mean=True,
                      data_dir=None,
                      save_dir=None,
                      **kwargs):

  qm7_tasks = ["u0_atom"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "qm7-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return qm7_tasks, all_dataset, transformers

  if featurizer == 'CoulombMatrix':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=QM7_MAT_URL, dest_dir=data_dir)

    dataset = scipy.io.loadmat(dataset_file)
    X = dataset['X']
    y = dataset['T'].T
    w = np.ones_like(y)
    dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)
  elif featurizer == 'BPSymmetryFunctionInput':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=QM7_MAT_URL, dest_dir=data_dir)
    dataset = scipy.io.loadmat(dataset_file)
    X = np.concatenate([np.expand_dims(dataset['Z'], 2), dataset['R']], axis=2)
    y = dataset['T'].reshape(-1, 1)  # scipy.io.loadmat puts samples on axis 1
    w = np.ones_like(y)
    dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)
  else:
    dataset_file = os.path.join(data_dir, "qm7.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=QM7_CSV_URL, dest_dir=data_dir)
    if featurizer == 'ECFP':
      featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
      featurizer = deepchem.feat.RawFeaturizer()
    elif featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      img_size = kwargs.get("img_size", 80)
      featurizer = deepchem.feat.SmilesToImage(
          img_size=img_size, img_spec=img_spec)
    loader = deepchem.data.CSVLoader(
        tasks=qm7_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file)

  if split == None:
    raise ValueError()
  else:
    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'stratified':
        deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
    }

    splitter = splitters[split]
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)

    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]

    for transformer in transformers:
      train_dataset = transformer.transform(train_dataset)
      valid_dataset = transformer.transform(valid_dataset)
      test_dataset = transformer.transform(test_dataset)
    if reload:
      deepchem.utils.save.save_dataset_to_disk(
          save_folder, train_dataset, valid_dataset, test_dataset, transformers)

    return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7b_from_mat(featurizer='CoulombMatrix',
                       split='stratified',
                       reload=True,
                       move_mean=True,
                       data_dir=None,
                       save_dir=None,
                       **kwargs):
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR
  dataset_file = os.path.join(data_dir, "qm7b.mat")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=QM7B_MAT_URL, dest_dir=data_dir)
  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)

  if split == None:
    raise ValueError()
  else:
    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'stratified':
        deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
    }
    splitter = splitters[split]
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)

    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]

    for transformer in transformers:
      train_dataset = transformer.transform(train_dataset)
      valid_dataset = transformer.transform(valid_dataset)
      test_dataset = transformer.transform(test_dataset)

    qm7_tasks = np.arange(y.shape[1])
    return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7(featurizer='CoulombMatrix',
             split='random',
             reload=True,
             move_mean=True,
             data_dir=None,
             save_dir=None,
             **kwargs):
  """Load qm7 datasets."""
  # Featurize qm7 dataset
  logger.info("About to featurize qm7 dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR
  dataset_file = os.path.join(data_dir, "gdb7.sdf")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=GDB7_URL, dest_dir=data_dir)
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

  if split == None:
    raise ValueError()

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
          transform_y=True, dataset=train_dataset, move_mean=move_mean)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers
