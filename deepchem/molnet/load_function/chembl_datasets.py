"""
ChEMBL dataset loader.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import deepchem
import pickle
from deepchem.molnet.load_function.chembl_tasks import chembl_tasks


def load_chembl(shard_size=2000,
                featurizer="ECFP",
                set="5thresh",
                split="random",
                reload=True):

  save = False
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
    if reload:
      save = True
  else:
    data_dir = "/tmp"

  dataset_path = os.path.join(data_dir, "chembl_%s.csv.gz" % set)
  if not os.path.exists(dataset_path):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_5thresh.csv.gz'
    )
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_sparse.csv.gz'
    )
    os.system(
        'wget -P ' + os.path.join(data_dir, 'chembl_year_sets') +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_test.csv.gz'
    )
    os.system(
        'wget -P ' + os.path.join(data_dir, 'chembl_year_sets') +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_train.csv.gz'
    )
    os.system(
        'wget -P ' + os.path.join(data_dir, 'chembl_year_sets') +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_valid.csv.gz'
    )
    os.system(
        'wget -P ' + os.path.join(data_dir, 'chembl_year_sets') +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_test.csv.gz'
    )
    os.system(
        'wget -P ' + os.path.join(data_dir, 'chembl_year_sets') +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_train.csv.gz'
    )
    os.system(
        'wget -P ' + os.path.join(data_dir, 'chembl_year_sets') +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_valid.csv.gz'
    )

  print("About to load ChEMBL dataset.")
  if save:
    save_dir = os.path.join(data_dir, "chembl/" + featurizer + "/" + split)
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
      return chembl_tasks, all_dataset, transformers

  if split == "year":
    train_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_train.csv.gz" % set)
    valid_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_valid.csv.gz" % set)
    test_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_test.csv.gz" % set)

  # Featurize ChEMBL dataset
  print("About to featurize ChEMBL dataset.")
  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=chembl_tasks, smiles_field="smiles", featurizer=featurizer)

  if split == "year":
    print("Featurizing train datasets")
    train_dataset = loader.featurize(train_files, shard_size=shard_size)
    print("Featurizing valid datasets")
    valid_dataset = loader.featurize(valid_files, shard_size=shard_size)
    print("Featurizing test datasets")
    test_dataset = loader.featurize(test_files, shard_size=shard_size)
  else:
    dataset = loader.featurize(dataset_path, shard_size=shard_size)
  # Initialize transformers
  print("About to transform data")
  if split == "year":
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset)
    ]
    for transformer in transformers:
      train = transformer.transform(train_dataset)
      valid = transformer.transform(valid_dataset)
      test = transformer.transform(test_dataset)
  else:
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]
    for transformer in transformers:
      dataset = transformer.transform(dataset)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }

  if split in splitters:
    splitter = splitters[split]
    print("Performing new split.")
    train, valid, test = splitter.train_valid_test_split(dataset)
  if save:
    train.move(train_dir)
    valid.move(valid_dir)
    test.move(test_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), 'w') as f:
      pickle.dump(transformers, f)
  return chembl_tasks, (train, valid, test), transformers
