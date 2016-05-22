"""
MUV dataset loader.
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

def load_muv(base_dir, mode="regression", transform=True, reload=True):
  """Load MUV datasets. Does not do train/test split"""
  # Set some global variables up top
  reload = True
  verbosity = "high"
  model = "logistic"

  # Create some directories for analysis
  # The base_dir holds the results of all analysis
  #base_dir = "/scratch/users/rbharath/muv_multitask_analysis"
  if not reload:
    if os.path.exists(base_dir):
      shutil.rmtree(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  current_dir = os.path.dirname(os.path.realpath(__file__))
  #Make directories to store the raw and featurized datasets.
  feature_dir = os.path.join(base_dir, "features")
  samples_dir = os.path.join(base_dir, "samples")
  data_dir = os.path.join(base_dir, "dataset")

  # Load MUV dataset
  print("About to load MUV dataset.")
  dataset_file = os.path.join(
      current_dir, "../../datasets/muv.csv.gz")
  dataset = load_from_disk(dataset_file)
  print("Columns of dataset: %s" % str(dataset.columns.values))
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))

  # Featurize MUV dataset
  print("About to featurize MUV dataset.")
  featurizers = [CircularFingerprint(size=1024)]
  all_MUV_tasks = sorted(['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644',
                          'MUV-548', 'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712',
                          'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733', 'MUV-652',
                          'MUV-466', 'MUV-832'])

  featurizer = DataFeaturizer(tasks=all_MUV_tasks,
                              smiles_field="smiles",
                              compound_featurizers=featurizers,
                              verbosity=verbosity)
  featurized_samples = featurizer.featurize(
      dataset_file, feature_dir,
      samples_dir, shard_size=8192,
      reload=reload)

  dataset = Dataset(data_dir=data_dir, samples=featurized_samples, 
                    featurizers=featurizers, tasks=all_MUV_tasks,
                    verbosity=verbosity, reload=reload)

  # Initialize transformers 
  input_transformers = []
  output_transformers = []
  weight_transformers = [
      BalancingTransformer(transform_w=True, dataset=dataset)]
  transformers = input_transformers + output_transformers + weight_transformers
  if not reload:
    print("About to transform data")
    for transformer in transformers:
        transformer.transform(dataset)
  
  return all_MUV_tasks, featurized_samples, dataset, transformers
