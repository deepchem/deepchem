"""
Contains BACE data loading utilities. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import os
import deepchem
import tempfile, shutil
from deepchem.utils.save import load_from_disk
from deepchem.splits import SpecifiedSplitter
from deepchem.featurizers import UserDefinedFeaturizer 
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.datasets import Dataset
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import ClippingTransformer
from deepchem.hyperparameters import HyperparamOpt
from sklearn.ensemble import RandomForestRegressor
from deepchem.models.sklearn_models import SklearnModel
from deepchem.datasets.bace_features import user_specified_features
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.utils.evaluate import Evaluator

def load_bace(mode="regression", transform=True, split="20-80"):
  """Load BACE-1 dataset as regression/classification problem."""
  reload = True
  verbosity = "high"
  regen = False
  assert split in ["20-80", "80-20"]

  current_dir = os.path.dirname(os.path.realpath(__file__))
  if split == "20-80":
    dataset_file = os.path.join(
        current_dir, "../../datasets/desc_canvas_aug30.csv")
  elif split == "80-20":
    dataset_file = os.path.join(
        current_dir, "../../datasets/rev8020split_desc.csv")
  dataset = load_from_disk(dataset_file)
  num_display = 10
  pretty_columns = (
      "[" + ",".join(["'%s'" % column for column in
  dataset.columns.values[:num_display]])
      + ",...]")

  crystal_dataset_file = os.path.join(
      current_dir, "../../datasets/crystal_desc_canvas_aug30.csv")
  crystal_dataset = load_from_disk(crystal_dataset_file)

  print("Columns of dataset: %s" % pretty_columns)
  print("Number of examples in dataset: %s" % str(dataset.shape[0]))
  print("Number of examples in crystal dataset: %s" %
  str(crystal_dataset.shape[0]))

  #Make directories to store the raw and featurized datasets.
  base_dir = tempfile.mkdtemp()
  data_dir = os.path.join(base_dir, "dataset")
  train_dir = os.path.join(base_dir, "train_dataset")
  valid_dir = os.path.join(base_dir, "valid_dataset")
  test_dir = os.path.join(base_dir, "test_dataset")
  model_dir = os.path.join(base_dir, "model")
  crystal_dir = os.path.join(base_dir, "crystal")

  if mode == "regression":
    bace_tasks = ["pIC50"]
  elif mode == "classification":
    bace_tasks = ["Class"]
  else:
    raise ValueError("Unknown mode %s" % mode)
  featurizer = UserDefinedFeaturizer(user_specified_features)
  loader = DataFeaturizer(tasks=bace_tasks,
                              smiles_field="mol",
                              id_field="CID",
                              featurizer=featurizer)
  if not reload or not os.path.exists(data_dir):
    dataset = loader.featurize(dataset_file, data_dir)
    regen = True
  else:
    dataset = Dataset(data_dir, reload=True)
  if not reload or not os.path.exists(crystal_dir):
    crystal_dataset = loader.featurize(crystal_dataset_file, crystal_dir)
  else:
    crystal_dataset = Dataset(crystal_dir, reload=True)


  if (not reload or not os.path.exists(train_dir) or not os.path.exists(valid_dir)
      or not os.path.exists(test_dir)):
    regen = True
    splitter = SpecifiedSplitter(dataset_file, "Model", verbosity=verbosity)
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, train_dir, valid_dir, test_dir)
  else:
    train_dataset = Dataset(train_dir, reload=True)
    valid_dataset = Dataset(valid_dir, reload=True)
    test_dataset = Dataset(test_dir, reload=True)

  #NOTE THE RENAMING:
  if split == "20-80":
    valid_dataset, test_dataset = test_dataset, valid_dataset
  print("Number of compounds in train set")
  print(len(train_dataset))
  print("Number of compounds in validation set")
  print(len(valid_dataset))
  print("Number of compounds in test set")
  print(len(test_dataset))
  print("Number of compounds in crystal set")
  print(len(crystal_dataset))

  if transform and regen:
    input_transformers = [
        NormalizationTransformer(transform_X=True, dataset=train_dataset),
        ClippingTransformer(transform_X=True, dataset=train_dataset)]
    output_transformers = []
    if mode == "regression":
      output_transformers = [
        NormalizationTransformer(transform_y=True, dataset=train_dataset)]
    else:
      output_transformers = []
  else:
    input_transformers, output_transformers = [], []
  
  transformers = input_transformers + output_transformers
  for dataset in [train_dataset, valid_dataset, test_dataset, crystal_dataset]:
    for transformer in transformers:
        transformer.transform(dataset)

  return (bace_tasks, train_dataset, valid_dataset, test_dataset,
          crystal_dataset, output_transformers)
