"""
NCI dataset loader.
Original Author - Bharath Ramsundar
Author - Aneesh Pappu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import pickle


def load_nci(featurizer='ECFP', shard_size=1000, split='random', reload=True):

  # Load nci dataset
  print("About to load NCI dataset.")
  save = False
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
    if reload:
      save = True
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "nci_unique.csv")
  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/nci_unique.csv'
    )

  all_nci_tasks = ([
      'CCRF-CEM', 'HL-60(TB)', 'K-562', 'MOLT-4', 'RPMI-8226', 'SR',
      'A549/ATCC', 'EKVX', 'HOP-62', 'HOP-92', 'NCI-H226', 'NCI-H23',
      'NCI-H322M', 'NCI-H460', 'NCI-H522', 'COLO 205', 'HCC-2998', 'HCT-116',
      'HCT-15', 'HT29', 'KM12', 'SW-620', 'SF-268', 'SF-295', 'SF-539',
      'SNB-19', 'SNB-75', 'U251', 'LOX IMVI', 'MALME-3M', 'M14', 'MDA-MB-435',
      'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'UACC-257', 'UACC-62', 'IGR-OV1',
      'OVCAR-3', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'NCI/ADR-RES', 'SK-OV-3',
      '786-0', 'A498', 'ACHN', 'CAKI-1', 'RXF 393', 'SN12C', 'TK-10', 'UO-31',
      'PC-3', 'DU-145', 'MCF7', 'MDA-MB-231/ATCC', 'MDA-MB-468', 'HS 578T',
      'BT-549', 'T-47D'
  ])

  if save:
    save_dir = os.path.join(data_dir, "nci/" + featurizer + "/" + split)
    train_dir = os.path.join(save_dir, "train_dir")
    valid_dir = os.path.join(save_dir, "valid_dir")
    test_dir = os.path.join(save_dir, "test_dir")
    if os.path.exists(train_dir) and os.path.exists(
        valid_dir) and os.path.exists(test_dir):
      train = deepchem.data.DiskDataset(train_dir)
      valid = deepchem.data.DiskDataset(valid_dir)
      test = deepchem.data.DiskDataset(test_dir)
      all_dataset = (train, valid, test)
      with open(os.path.join(save_dir, "transformers.pkl"), 'r') as f:
        transformers = pickle.load(f)
      return all_nci_tasks, all_dataset, transformers

  # Featurize nci dataset
  print("About to featurize nci dataset.")
  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=all_nci_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file, shard_size=shard_size)

  # Initialize transformers
  print("About to transform data")
  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=dataset)
  ]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  print("Performing new split.")
  train, valid, test = splitter.train_valid_test_split(dataset)
  if save:
    train.move(train_dir)
    valid.move(valid_dir)
    test.move(test_dir)
    with open(os.path.join(save_dir, "transformers.pkl"), 'w') as f:
      pickle.dump(transformers, f)
  return all_nci_tasks, (train, valid, test), transformers
