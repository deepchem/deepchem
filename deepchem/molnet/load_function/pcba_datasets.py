"""
PCBA dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import gzip


def load_pcba(featurizer='ECFP', split='random', reload=True):
  return load_pcba_dataset(
      featurizer=featurizer,
      split=split,
      reload=reload,
      assay_file_name="pcba.csv.gz")


def load_pcba_146(featurizer='ECFP', split='random', reload=True):
  return load_pcba_dataset(
      featurizer=featurizer,
      split=split,
      reload=reload,
      assay_file_name="pcba_146.csv.gz")


def load_pcba_2475(featurizer='ECFP', split='random', reload=True):
  return load_pcba_dataset(
      featurizer=featurizer,
      split=split,
      reload=reload,
      assay_file_name="pcba_2475.csv.gz")


def load_pcba_dataset(featurizer='ECFP',
                      split='random',
                      reload=True,
                      assay_file_name="pcba.csv.gz"):
  """Load PCBA datasets. Does not do train/test split"""
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(
        data_dir, assay_file_name.split(".")[0] + featurizer + "/" + split)

  dataset_file = os.path.join(data_dir, assay_file_name)

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(
        "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/{0}".
        format(assay_file_name))

  # Featurize PCBA dataset
  print("About to featurize PCBA dataset.")
  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  with gzip.GzipFile(dataset_file, "r") as fin:
    header = fin.readline().rstrip().decode("utf-8")
    columns = header.split(",")
    columns.remove("mol_id")
    columns.remove("smiles")
    PCBA_tasks = columns

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return PCBA_tasks, all_dataset, transformers

  loader = deepchem.data.CSVLoader(
      tasks=PCBA_tasks, smiles_field="smiles", featurizer=featurizer)

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
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  print("Performing new split.")
  train, valid, test = splitter.train_valid_test_split(dataset)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)

  return PCBA_tasks, (train, valid, test), transformers
