"""
Clinical Toxicity (clintox) dataset loader.
@author Caleb Geniesse
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import pickle


def load_clintox(featurizer='ECFP', split='index', reload=True):
  """Load clintox datasets."""

  save = False
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
    if reload:
      save = True
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "clintox.csv.gz")
  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    )

  print("About to load clintox dataset.")
  dataset = deepchem.utils.save.load_from_disk(dataset_file)
  clintox_tasks = dataset.columns.values[1:].tolist()
  print("Tasks in dataset: %s" % (clintox_tasks))
  print("Number of tasks in dataset: %s" % str(len(clintox_tasks)))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))
  if save:
    save_dir = os.path.join(data_dir, "clintox/" + featurizer + "/" + split)
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
      return clintox_tasks, all_dataset, transformers
  # Featurize clintox dataset
  print("About to featurize clintox dataset.")
  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=clintox_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Transform clintox dataset
  print("About to transform clintox dataset.")
  transformers = [
      deepchem.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  # Split clintox dataset
  print("About to split clintox dataset.")
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

  return clintox_tasks, (train, valid, test), transformers
