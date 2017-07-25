"""
Load datasets for Low Data processing.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import shutil
import tempfile
import numpy as np
import deepchem as dc

def to_numpy_dataset(dataset):
  """Converts dataset to numpy dataset."""
  return dc.data.NumpyDataset(dataset.X, dataset.y, dataset.w, dataset.ids)

def load_tox21_ecfp(num_train=7200):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")
  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

  loader = dc.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  return tox21_tasks, dataset, transformers

def load_tox21_convmol(base_dir=None, num_train=7200):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")

  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  featurizer = dc.feat.ConvMolFeaturizer()
  tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                 'SR-HSE', 'SR-MMP', 'SR-p53']

  loader = dc.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  return tox21_tasks, dataset, transformers

def load_muv_ecfp():
  """Load MUV datasets. Does not do train/test split"""
  # Load MUV dataset
  print("About to load MUV dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../../datasets/muv.csv.gz")
  # Featurize MUV dataset
  print("About to featurize MUV dataset.")
  featurizer = dc.feat.CircularFingerprint(size=1024)
  MUV_tasks = sorted(['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644',
                      'MUV-548', 'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712',
                      'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733', 'MUV-652',
                      'MUV-466', 'MUV-832'])

  loader = dc.data.CSVLoader(
      tasks=MUV_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  return MUV_tasks, dataset, transformers

def load_muv_convmol():
  """Load MUV datasets. Does not do train/test split"""
  # Load MUV dataset
  print("About to load MUV dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../../datasets/muv.csv.gz")
  # Featurize MUV dataset
  print("About to featurize MUV dataset.")
  featurizer = dc.feat.ConvMolFeaturizer()
  MUV_tasks = sorted(['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644',
                      'MUV-548', 'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712',
                      'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733', 'MUV-652',
                      'MUV-466', 'MUV-832'])

  loader = dc.data.CSVLoader(
      tasks=MUV_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  return MUV_tasks, dataset, transformers

def load_sider_ecfp():
  """Load SIDER datasets. Does not do train/test split"""
  # Featurize SIDER dataset
  print("About to featurize SIDER dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../sider/sider.csv.gz")
  featurizer = dc.feat.CircularFingerprint(size=1024)

  dataset = dc.utils.save.load_from_disk(dataset_file)
  SIDER_tasks = dataset.columns.values[1:].tolist()
  print("SIDER tasks: %s" % str(SIDER_tasks))
  print("%d tasks in total" % len(SIDER_tasks))


  loader = dc.data.CSVLoader(
      tasks=SIDER_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  print("%d datapoints in SIDER dataset" % len(dataset))

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  return SIDER_tasks, dataset, transformers

def load_sider_convmol():
  """Load SIDER datasets. Does not do train/test split"""
  # Featurize SIDER dataset
  print("About to featurize SIDER dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "../sider/sider.csv.gz")
  featurizer = dc.feat.ConvMolFeaturizer()

  dataset = dc.utils.save.load_from_disk(dataset_file)
  SIDER_tasks = dataset.columns.values[1:].tolist()
  print("SIDER tasks: %s" % str(SIDER_tasks))
  print("%d tasks in total" % len(SIDER_tasks))


  loader = dc.data.CSVLoader(
      tasks=SIDER_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  print("%d datapoints in SIDER dataset" % len(dataset))

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  return SIDER_tasks, dataset, transformers
