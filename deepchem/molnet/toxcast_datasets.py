"""
TOXCAST dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc

def load_toxcast(featurizer='ECFP', split='index'):

  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(
      data_dir, "./toxcast_data.csv.gz")
  if not os.path.exists(dataset_file):
    os.system('wget -P ' + data_dir + 
    ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz')
    
  dataset = dc.utils.save.load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize TOXCAST dataset
  print("About to featurize TOXCAST dataset.")

  featurizers = {'ECFP': dc.feat.CircularFingerprint(size=1024),
                 'GraphConv': dc.feat.ConvMolFeaturizer()}
  featurizer = featurizers[featurizer]

  TOXCAST_tasks = dataset.columns.values[1:].tolist()

  loader = dc.data.CSVLoader(
      tasks=TOXCAST_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter()}
  splitter = splitters[split]

  train, valid, test = splitter.train_valid_test_split(dataset)
  
  return TOXCAST_tasks, (train, valid, test), transformers
