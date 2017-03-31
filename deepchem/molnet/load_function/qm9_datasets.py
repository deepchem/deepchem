"""
qm9 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import pickle


def load_qm9(featurizer=None, split='random', reload=True):
  """Load qm9 datasets."""
  # Featurize qm9 dataset
  print("About to featurize qm9 dataset.")
  save = False
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
    if reload:
      save = True
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "gdb9.sdf")

  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz '
    )
    os.system('tar -zxvf ' + os.path.join(data_dir, 'gdb9.tar.gz') + ' -C ' +
              data_dir)

  qm9_tasks = [
      "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv",
      "u0_atom", "u298_atom", "h298_atom", "g298_atom"
  ]

  if save:
    save_dir = os.path.join(data_dir, "qm9/" + featurizer + "/" + split)
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
      return qm9_tasks, all_dataset, transformers

  if featurizer is None:
    featurizer = deepchem.feat.CoulombMatrix(29)
  loader = deepchem.data.SDFLoader(
      tasks=qm9_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=11)
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
  return qm9_tasks, (train_dataset, valid_dataset, test_dataset), transformers
