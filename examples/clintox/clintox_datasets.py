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

  # Load clintox dataset
  print("About to load clintox dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "./datasets/clintox.csv.gz")
  dataset = dc.utils.save.load_from_disk(dataset_file)
  clintox_tasks = dataset.columns.values[1:].tolist()
  print("Tasks in dataset: %s" % (clintox_tasks))
  print("Number of tasks in dataset: %s" % str(len(clintox_tasks)))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize clintox dataset
  print("About to featurize clintox dataset.")
  featurizers = {'ECFP': dc.feat.CircularFingerprint(size=1024),
                 'GraphConv': dc.feat.ConvMolFeaturizer()}
  featurizer = featurizers[featurizer]
  loader = dc.data.CSVLoader(
      tasks=clintox_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Transform clintox dataset
  print("About to transform clintox dataset.")
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  # Split clintox dataset
  print("About to split clintox dataset.")
  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter()}
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  return clintox_tasks, (train, valid, test), transformers
