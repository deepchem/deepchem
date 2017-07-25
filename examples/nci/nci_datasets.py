"""
NCI dataset loader.
Original Author - Bharath Ramsundar
Author - Aneesh Pappu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import numpy as np
import shutil
import deepchem as dc

def load_nci(featurizer='ECFP', shard_size=1000, split='random'):

  current_dir = os.path.dirname(os.path.realpath(__file__))

  # Load nci dataset
  print("About to load NCI dataset.")
  dataset_path = os.path.join(
      current_dir, "../../datasets/nci_unique.csv")


  # Featurize nci dataset
  print("About to featurize nci dataset.")
  if featurizer == 'ECFP':
      featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
      featurizer = dc.feat.ConvMolFeaturizer()

  all_nci_tasks = (['CCRF-CEM', 'HL-60(TB)', 'K-562', 'MOLT-4', 'RPMI-8226',
                    'SR', 'A549/ATCC', 'EKVX', 'HOP-62', 'HOP-92', 'NCI-H226',
                    'NCI-H23', 'NCI-H322M', 'NCI-H460', 'NCI-H522', 'COLO 205',
                    'HCC-2998', 'HCT-116', 'HCT-15', 'HT29', 'KM12', 'SW-620',
                    'SF-268', 'SF-295', 'SF-539', 'SNB-19', 'SNB-75', 'U251',
                    'LOX IMVI', 'MALME-3M', 'M14', 'MDA-MB-435', 'SK-MEL-2',
                    'SK-MEL-28', 'SK-MEL-5', 'UACC-257', 'UACC-62', 'IGR-OV1',
                    'OVCAR-3', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'NCI/ADR-RES',
                    'SK-OV-3', '786-0', 'A498', 'ACHN', 'CAKI-1', 'RXF 393',
                    'SN12C', 'TK-10', 'UO-31', 'PC-3', 'DU-145', 'MCF7',
                    'MDA-MB-231/ATCC', 'MDA-MB-468', 'HS 578T', 'BT-549',
                    'T-47D'])

  loader = dc.data.CSVLoader(
      tasks=all_nci_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_path, shard_size=shard_size)

  # Initialize transformers
  print("About to transform data")
  transformers = [
      dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]
  for transformer in transformers:
    dataset = transformer.transform(dataset)
  
  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter()}
  splitter = splitters[split]
  print("Performing new split.")
  train, valid, test = splitter.train_valid_test_split(dataset)

  return all_nci_tasks, (train, valid, test), transformers
