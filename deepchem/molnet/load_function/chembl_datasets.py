"""
ChEMBL dataset loader.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import deepchem
from deepchem.molnet.load_function.chembl_tasks import chembl_tasks


def load_chembl(shard_size=2000,
                featurizer="ECFP",
                set="5thresh",
                split="random",
                reload=True):

  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "chembl/" + featurizer + "/" + split)

  dataset_path = os.path.join(data_dir, "chembl_%s.csv.gz" % set)
  if not os.path.exists(dataset_path):
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_5thresh.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_sparse.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_test.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_train.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_valid.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_test.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_train.csv.gz'
    )
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_valid.csv.gz'
    )

  print("About to load ChEMBL dataset.")
  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
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
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
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

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return chembl_tasks, (train, valid, test), transformers
