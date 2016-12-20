"""
Contains BACE data loading utilities. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import os
import deepchem
import tempfile
import deepchem as dc

def load_bace(mode="regression", transform=True, split="20-80"):
  """Load BACE-1 dataset as regression/classification problem."""
  assert split in ["20-80", "80-20"]
  assert mode in ["regression", "classification"]

  current_dir = os.path.dirname(os.path.realpath(__file__))
  if split == "20-80":
    dataset_file = os.path.join(
        current_dir, "../../datasets/desc_canvas_aug30.csv")
  elif split == "80-20":
    dataset_file = os.path.join(
        current_dir, "../../datasets/rev8020split_desc.csv")

  crystal_dataset_file = os.path.join(
      current_dir, "../../datasets/crystal_desc_canvas_aug30.csv")

  if mode == "regression":
    bace_tasks = ["pIC50"]
  elif mode == "classification":
    bace_tasks = ["Class"]
  featurizer = dc.feat.UserDefinedFeaturizer(user_specified_features)
  loader = dc.data.UserCSVLoader(
      tasks=bace_tasks, smiles_field="mol", id_field="CID",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  crystal_dataset = loader.featurize(crystal_dataset_file)

  splitter = dc.splits.SpecifiedSplitter(dataset_file, "Model")
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

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

  transformers = [
      NormalizationTransformer(transform_X=True, dataset=train_dataset),
      ClippingTransformer(transform_X=True, dataset=train_dataset)]
  if mode == "regression":
    transformers += [
      NormalizationTransformer(transform_y=True, dataset=train_dataset)]
  
  for dataset in [train_dataset, valid_dataset, test_dataset, crystal_dataset]:
    for transformer in transformers:
        dataset = transformer.transform(dataset)

  return (bace_tasks, (train_dataset, valid_dataset, test_dataset,
          crystal_dataset), transformers)
