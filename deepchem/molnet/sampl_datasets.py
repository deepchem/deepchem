"""
SAMPL dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc

def load_sampl(featurizer='ECFP', split='index'):
  """Load SAMPL datasets."""
  # Featurize SAMPL dataset
  print("About to featurize SAMPL dataset.")
  print("About to load MUV dataset.")
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"
  
  dataset_file = os.path.join(
      data_dir, "./SAMPL.csv")
  if not os.path.exists(dataset_file):
    os.system('wget -P ' + data_dir + 
    ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/SAMPL.csv')
    
  SAMPL_tasks = ['expt']
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  loader = dc.data.CSVLoader(
      tasks=SAMPL_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(
      dataset_file, shard_size=8192)

  # Initialize transformers 
  transformers = [
      dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter()}
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return SAMPL_tasks, (train, valid, test), transformers
