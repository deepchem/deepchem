"""
HOPV dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem


def load_hopv(featurizer='ECFP', split='index'):
  """Load HOPV datasets. Does not do train/test split"""
  # Featurize HOPV dataset
  print("About to featurize HOPV dataset.")
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "hopv.csv")
  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/hopv.tar.gz'
    )
    os.system('tar -zxvf ' + os.path.join(data_dir, 'hopv.tar.gz') + ' -C ' +
              data_dir)

  hopv_tasks = [
      'HOMO', 'LUMO', 'electrochemical_gap', 'optical_gap', 'PCE', 'V_OC',
      'J_SC', 'fill_factor'
  ]
  if featurizer == 'ECFP':
    featurizer_func = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer_func = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=hopv_tasks, smiles_field="smiles", featurizer=featurizer_func)
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
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return hopv_tasks, (train, valid, test), transformers
