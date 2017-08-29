"""
Lipophilicity dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem


def load_lipo(featurizer='ECFP', split='index', reload=True):
  """Load Lipophilicity datasets."""
  # Featurize Lipophilicity dataset
  print("About to featurize Lipophilicity dataset.")
  print("About to load Lipophilicity dataset.")
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "lipo/" + featurizer + "/" + split)

  dataset_file = os.path.join(data_dir, "Lipophilicity.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(
        'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
    )

  Lipo_tasks = ['exp']

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return Lipo_tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=Lipo_tasks, smiles_field="smiles", featurizer=featurizer)
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

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return Lipo_tasks, (train, valid, test), transformers
