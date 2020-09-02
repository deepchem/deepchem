"""
Testing singletask/multitask dataset merging
"""
import os
import deepchem as dc
import numpy as np


def test_merge():
  """Test that datasets can be merged."""
  current_dir = os.path.dirname(os.path.realpath(__file__))

  dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["log-solubility"]
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  first_dataset = loader.create_dataset(dataset_file)
  second_dataset = loader.create_dataset(dataset_file)

  merged_dataset = dc.data.DiskDataset.merge([first_dataset, second_dataset])

  assert len(merged_dataset) == len(first_dataset) + len(second_dataset)


def test_subset():
  """Tests that subsetting of datasets works."""
  current_dir = os.path.dirname(os.path.realpath(__file__))

  dataset_file = os.path.join(current_dir, "../../models/tests/example.csv")

  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["log-solubility"]
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(dataset_file, shard_size=2)

  shard_nums = [1, 2]

  orig_ids = dataset.ids
  _, _, _, ids_1 = dataset.get_shard(1)
  _, _, _, ids_2 = dataset.get_shard(2)

  subset = dataset.subset(shard_nums)
  after_ids = dataset.ids

  assert len(subset) == 4
  assert sorted(subset.ids) == sorted(np.concatenate([ids_1, ids_2]))
  assert list(orig_ids) == list(after_ids)
