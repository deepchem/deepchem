"""
Sider dataset loader.
@Author Bharath Ramsundar, Aneesh Pappu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
from deepchem.utils.save import load_from_disk
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataLoader
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.transformers import BalancingTransformer

def load_sider(base_dir, reload=True):
  """Load Sider datasets. Does not do train/test split"""
  # Set some global variables up top
  verbosity = "high"
  model = "logistic"
  regen = False

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if not reload:
    if os.path.exists(base_dir):
      shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  # Make directories to store the raw and featurized datasets.
  data_dir = os.path.join(base_dir, "dataset")

  # Load sider dataset
  print("About to load sider dataset.")
  dataset_file = os.path.join(
    current_dir, "./sider.csv.gz")
  dataset = load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize sider dataset
  print("About to featurize siderdataset.")

  featurizer = CircularFingerprint(size=1024)
  all_sider_tasks = dataset.columns.values[1:].tolist()

  if not reload or not os.path.exists(data_dir):
    loader = DataLoader(tasks=all_sider_tasks,
                        smiles_field="smiles",
                        featurizer=featurizer,
                        verbosity=verbosity)
    dataset = loader.featurize(
      dataset_file, data_dir, shard_size=8192)
  else:
    dataset = Dataset(data_dir, all_sider_tasks, reload=True)

  # Initialize transformers
  transformers = [
    BalancingTransformer(transform_w=True, dataset=dataset)]
  if regen:
    print("About to transform data")
    for transformer in transformers:
      transformer.transform(dataset)

  return all_sider_tasks, dataset, transformers
