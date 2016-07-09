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
from deepchem.utils.save import load_sharded_csv
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import NormalizationTransformer

def load_nci(base_dir, reload=True, force_transform=False):
  """Load NCI datasets. Does not do train/test split"""
  # Set some global variables up top
  verbosity = "high"
  model = "logistic"
  regen = False

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if not reload:
    if os.path.exists(base_dir):
      print("Deleting dir in nci_datasets.py")
      print(base_dir)
      shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.
  data_dir = os.path.join(base_dir, "dataset")

  # Load nci dataset
  print("About to load NCI dataset.")
  dataset_file1_path = os.path.join(
      current_dir, "../../datasets/nci_1.csv.gz")
  dataset_file2_path = os.path.join(
      current_dir, "../../datasets/nci_2.csv.gz")
  dataset_paths = [dataset_file1_path, dataset_file2_path]
  dataset = load_sharded_csv(dataset_paths)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize nci dataset
  print("About to featurize nci dataset.")
  featurizers = [CircularFingerprint(size=1024)]
  #was sorted list originally in muv_datasets.py, but csv is ordered so removed
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

  featurizer = DataFeaturizer(tasks=all_nci_tasks,
                              smiles_field="smiles",
                              featurizers=featurizers,
                              verbosity=verbosity)
  if not reload or not os.path.exists(data_dir):
    dataset = featurizer.featurize(dataset_paths, data_dir)
    regen = True
  else:
    dataset = Dataset(data_dir, reload=True)

  # Initialize transformers
  if regen or force_transform:
    print("About to transform data")
    transformers = [
        NormalizationTransformer(transform_y=True, dataset=dataset)]
    for transformer in transformers:
        transformer.transform(dataset)

  return all_nci_tasks, dataset, transformers
