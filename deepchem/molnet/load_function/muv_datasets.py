"""
MUV dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import pickle


def load_muv(featurizer='ECFP', split='index', reload=True, K=4):
  """Load MUV datasets. Does not do train/test split"""
  # Load MUV dataset
  print("About to load MUV dataset.")
  save = False
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
    if reload:
      save = True
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "muv.csv.gz")
  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/muv.csv.gz'
    )

  MUV_tasks = sorted([
      'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548',
      'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858',
      'MUV-713', 'MUV-733', 'MUV-652', 'MUV-466', 'MUV-832'
  ])

  if save:
    save_dir = os.path.join(data_dir, "muv/" + featurizer + "/" + split)
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
      return MUV_tasks, all_dataset, transformers

  # Featurize MUV dataset
  print("About to featurize MUV dataset.")

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=MUV_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  # Initialize transformers 
  transformers = [
      deepchem.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'task': deepchem.splits.TaskSplitter()
  }
  splitter = splitters[split]
  if split == 'task':
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
  else:
    train, valid, test = splitter.train_valid_test_split(dataset)
    if save:
      train.move(train_dir)
      valid.move(valid_dir)
      test.move(test_dir)
      with open(os.path.join(save_dir, "transformers.pkl"), 'w') as f:
        pickle.dump(transformers, f)
    all_dataset = (train, valid, test)
  return MUV_tasks, all_dataset, transformers
