"""
SuperToxic (STox) dataset loader.
@author Caleb Geniesse
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc


def load_supertoxic(featurizer='ECFP', split='index'):
  """Load supertoxic datasets."""

  # Load supertoxic dataset
  print("About to load supertoxic dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "./datasets/stox_processed-canonicalized-dense10K.csv.gz")
  dataset = dc.utils.save.load_from_disk(dataset_file)
  supertoxic_tasks = dataset.columns.values[1:].tolist()
  
  print("Dataset file: %s" % (dataset_file))
  print("Tasks in dataset: %s" % (supertoxic_tasks))
  print("Number of tasks in dataset: %s" % str(len(supertoxic_tasks)))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize supertoxic dataset
  print("About to featurize supertoxic dataset.")
  featurizers = {'ECFP': dc.feat.CircularFingerprint(size=1024),
                 'GraphConv': dc.feat.ConvMolFeaturizer()}
  if featurizer in featurizers:
    featurizer = featurizers[featurizer]
  loader = dc.data.CSVLoader(
      tasks=supertoxic_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Transform supertoxic dataset
  print("About to transform supertoxic dataset.")
  transformers = [
      dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  # Split supertoxic dataset
  print("About to split supertoxic dataset.")
  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter()}
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  return supertoxic_tasks, (train, valid, test), transformers
