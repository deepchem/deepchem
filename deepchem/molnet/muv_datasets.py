"""
MUV dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc

def load_muv(featurizer='ECFP', split='index'):
  """Load MUV datasets. Does not do train/test split"""
  # Load MUV dataset
  print("About to load MUV dataset.")
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"
  
  dataset_file = os.path.join(
      data_dir, "./muv.csv.gz")
  if not os.path.exists(dataset_file):
    os.system('wget -P ' + data_dir + 
    ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/muv.csv.gz')
    
  # Featurize MUV dataset
  print("About to featurize MUV dataset.")

  featurizers = {'ECFP': dc.feat.CircularFingerprint(size=1024),
                 'GraphConv': dc.feat.ConvMolFeaturizer()}
  featurizer = featurizers[featurizer]
      
  MUV_tasks = sorted(['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644',
                      'MUV-548', 'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712',
                      'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733', 'MUV-652',
                      'MUV-466', 'MUV-832'])

  loader = dc.data.CSVLoader(
      tasks=MUV_tasks, smiles_field="smiles", featurizer=featurizer)
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
  return MUV_tasks, (train, valid, test), transformers



