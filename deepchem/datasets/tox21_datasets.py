"""
Tox21 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from deepchem.utils.save import load_from_disk
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import RandomSplitter
from deepchem.datasets import Dataset
from deepchem.transformers import BalancingTransformer
from deepchem.hyperparameters import HyperparamOpt
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.metrics import to_one_hot
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import relative_difference
from deepchem.utils.evaluate import Evaluator

def load_tox21(base_dir, reload=True):
  """Load Tox21 datasets. Does not do train/test split"""
  # Set some global variables up top
  reload = True
  verbosity = "high"
  model = "logistic"

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  if not reload:
    if os.path.exists(base_dir):
      shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.
  samples_dir = os.path.join(base_dir, "samples")
  data_dir = os.path.join(base_dir, "dataset")

  # Load Tox21 dataset
  print("About to load Tox21 dataset.")
  dataset_file = os.path.join(
      current_dir, "../../datasets/tox21.csv.gz")
  dataset = load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  featurizers = [CircularFingerprint(size=1024)]
  all_tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                     'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                     'SR-HSE', 'SR-MMP', 'SR-p53']

  if not reload or not os.path.exists(data_dir):
    featurizer = DataFeaturizer(tasks=all_tox21_tasks,
                                smiles_field="smiles",
                                featurizers=featurizers,
                                verbosity=verbosity)
    dataset = featurizer.featurize(
        dataset_file, data_dir, shard_size=8192)
  else:
    dataset = Dataset(data_dir, all_tox21_tasks, reload=True)

  # Initialize transformers 
  transformers = [
      BalancingTransformer(transform_w=True, dataset=dataset)]
  if not reload:
    print("About to transform data")
    for transformer in transformers:
        transformer.transform(dataset)
  
  return all_tox21_tasks, dataset, transformers
