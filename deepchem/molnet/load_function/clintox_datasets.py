"""
Clinical Toxicity (clintox) dataset loader.
@author Caleb Geniesse
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc


def load_clintox(featurizer='ECFP', split='index'):
  """Load clintox datasets."""

  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "clintox.csv.gz")
  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    )

  print("About to load clintox dataset.")
  dataset = dc.utils.save.load_from_disk(dataset_file)
  clintox_tasks = dataset.columns.values[1:].tolist()
  print("Tasks in dataset: %s" % (clintox_tasks))
  print("Number of tasks in dataset: %s" % str(len(clintox_tasks)))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize clintox dataset
  print("About to featurize clintox dataset.")
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = dc.feat.RawFeaturizer()

  loader = dc.data.CSVLoader(
      tasks=clintox_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Transform clintox dataset
  print("About to transform clintox dataset.")
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  # Split clintox dataset
  print("About to split clintox dataset.")
  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  return clintox_tasks, (train, valid, test), transformers
