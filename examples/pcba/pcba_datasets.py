"""
PCBA dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc


def load_pcba(featurizer='ECFP', split='random'):
  """Load PCBA datasets."""

  current_dir = os.path.dirname(os.path.realpath(__file__))
  print("About to load PCBA dataset.")
  dataset_file = os.path.join(current_dir, "../../datasets/pcba.csv.gz")

  # Featurize PCBA dataset
  print("About to featurize PCBA dataset.")
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  PCBA_tasks = [
      'PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457',
      'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469',
      'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688',
      'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242',
      'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546',
      'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676',
      'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294',
      'PCBA-485297', 'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349',
      'PCBA-485353', 'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947',
      'PCBA-493208', 'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339',
      'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842',
      'PCBA-504845', 'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317',
      'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590',
      'PCBA-588591', 'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233',
      'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171',
      'PCBA-624173', 'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288',
      'PCBA-624291', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635',
      'PCBA-651644', 'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104',
      'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979',
      'PCBA-720504', 'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553',
      'PCBA-720579', 'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709',
      'PCBA-720711', 'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881',
      'PCBA-883', 'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899',
      'PCBA-902', 'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915',
      'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995'
  ]

  loader = dc.data.CSVLoader(
      tasks=PCBA_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file)
  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  print("Performing new split.")
  train, valid, test = splitter.train_valid_test_split(dataset)

  return PCBA_tasks, (train, valid, test), transformers
