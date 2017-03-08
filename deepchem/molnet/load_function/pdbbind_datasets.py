"""
PDBBind dataset loader.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import multiprocessing
import os
import re
import time

import deepchem as dc
import numpy as np
import pandas as pd

def featurize_pdbbind(data_dir=None, feat="grid", subset="core"):
  """Featurizes pdbbind according to provided featurization"""
  tasks = ["-logKd/Ki"]
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"
  data_dir = os.path.join(data_dir, "pdbbind")
  dataset_dir = os.path.join(data_dir, "%s_%s" % (subset, feat))
  
  if not os.path.exists(dataset_dir):
    os.system('wget -P ' + data_dir + 
    ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz')
    os.system('wget -P ' + data_dir + 
    ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/full_grid.tar.gz')
    os.system('wget -P ' + data_dir + 
    ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/refined_grid.tar.gz')
    os.system('tar -zxvf ' + os.path.join(data_dir, 'core_grid.tar.gz') + 
    ' -C ' + data_dir)
    os.system('tar -zxvf ' + os.path.join(data_dir, 'full_grid.tar.gz') + 
    ' -C ' + data_dir)
    os.system('tar -zxvf ' + os.path.join(data_dir, 'refined_grid.tar.gz') + 
    ' -C ' + data_dir)

  return dc.data.DiskDataset(dataset_dir), tasks


def load_pdbbind_grid(split="index", featurizer="grid", subset="full"):
  """Load PDBBind datasets. Does not do train/test split"""
  dataset, tasks = featurize_pdbbind(feat=featurizer, subset=subset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  transformers = []
  for transformer in transformers:
    train = transformer.transform(train)
  for transformer in transformers:
    valid = transformer.transform(valid)
  for transformer in transformers:
    test = transformer.transform(test)

  return tasks, (train, valid, test), transformers
