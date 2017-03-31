"""
Delaney dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import pickle


def load_delaney(featurizer='ECFP', split='index', reload=True):
  """Load delaney datasets."""
  # Featurize Delaney dataset
  print("About to featurize Delaney dataset.")
  save = False
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
    if reload:
      save = True
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "delaney-processed.csv")

  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
    )

  delaney_tasks = ['measured log solubility in mols per litre']
  if save:
    save_dir = os.path.join(data_dir, "delaney/" + featurizer + "/" + split)
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
      return delaney_tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=delaney_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  if save:
    train.move(train_dir)
    valid.move(valid_dir)
    test.move(test_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), 'w') as f:
      pickle.dump(transformers, f)
  return delaney_tasks, (train, valid, test), transformers
