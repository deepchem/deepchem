"""
qm8 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc


def load_qm8(featurizer=None, split='random'):
  """Load qm8 datasets."""
  # Featurize qm8 dataset
  print("About to featurize qm8 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir, "./qm8.sdf")
  if not os.path.exists(dataset_file):
    os.system('sh ' + current_dir + '/get_qm8.sh')

  qm8_tasks = [
      "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
      "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM",
      "f1-CAM", "f2-CAM"
  ]
  if featurizer is None:
    featurizer = dc.feat.CoulombMatrix(26)
  loader = dc.data.SDFLoader(
      tasks=qm8_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
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
  return qm8_tasks, (train_dataset, valid_dataset, test_dataset), transformers
