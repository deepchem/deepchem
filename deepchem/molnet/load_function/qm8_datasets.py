"""
qm8 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import pickle


def load_qm8(featurizer=None, split='random', reload=True):
  save = False
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
    if reload:
      save = True
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "qm8.sdf")

  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz '
    )
    os.system('tar -zxvf ' + os.path.join(data_dir, 'gdb8.tar.gz') + ' -C ' +
              data_dir)

  qm8_tasks = [
      "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
      "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM",
      "f1-CAM", "f2-CAM"
  ]

  if save:
    save_dir = os.path.join(data_dir, "qm8/" + featurizer + "/" + split)
    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    if os.path.exists(train_dir) and os.path.exists(
        valid_dir) and os.path.exists(test_dir):
      train = deepchem.data.DiskDataset(train_dir)
      valid = deepchem.data.DiskDataset(valid_dir)
      test = deepchem.data.DiskDataset(test_dir)
      all_dataset = (train, valid, test)
      with open(os.path.join(save_dir, "transformers.pkl"), 'r') as f:
        transformers = pickle.load(f)
      return qm8_tasks, all_dataset, transformers

  if featurizer is None:
    featurizer = deepchem.feat.CoulombMatrix(26)
  loader = deepchem.data.SDFLoader(
      tasks=qm8_tasks,
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
  if save:
    train_dataset.move(train_dir)
    valid_dataset.move(valid_dir)
    test_dataset.move(test_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), 'w') as f:
      pickle.dump(transformers, f)
  return qm8_tasks, (train_dataset, valid_dataset, test_dataset), transformers
