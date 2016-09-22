"""
Tests for splitter objects.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import tempfile
import numpy as np
from deepchem.datasets import Dataset
from deepchem.splits import RandomSplitter
from deepchem.splits import IndexSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import StratifiedSplitter
from deepchem.datasets.tests import TestDatasetAPI


class TestSplitters(TestDatasetAPI):
  """
  Test some basic splitters.
  """

  def test_singletask_random_split(self):
    """
    Test singletask RandomSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    random_splitter = RandomSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            solubility_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

    merge_dir = tempfile.mkdtemp()
    merged_dataset = Dataset.merge(
        merge_dir, [train_data, valid_data, test_data])
    assert sorted(merged_dataset.get_ids()) == (
           sorted(solubility_dataset.get_ids()))

  def test_singletask_index_split(self):
    """
    Test singletask RandomSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    random_splitter = IndexSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            solubility_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

    merge_dir = tempfile.mkdtemp()
    merged_dataset = Dataset.merge(
        merge_dir, [train_data, valid_data, test_data])
    assert sorted(merged_dataset.get_ids()) == (
           sorted(solubility_dataset.get_ids()))

  def test_singletask_scaffold_split(self):
    """
    Test singletask ScaffoldSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    scaffold_splitter = ScaffoldSplitter()
    train_data, valid_data, test_data = \
        scaffold_splitter.train_valid_test_split(
            solubility_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_singletask_random_k_fold_split(self):
    """
    Test singletask RandomSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    random_splitter = RandomSplitter()
    ids_set = set(solubility_dataset.get_ids())

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = random_splitter.k_fold_split(solubility_dataset, fold_dirs)
    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      # Verify lengths is 10/k == 2
      assert len(fold_dataset) == 2
      # Verify that compounds in this fold are subset of original compounds
      fold_ids_set = set(fold_dataset.get_ids())
      assert fold_ids_set.issubset(ids_set)
      # Verify that no two folds have overlapping compounds.
      for other_fold in range(K):
        if fold == other_fold:
          continue
        other_fold_dataset = fold_datasets[other_fold]
        other_fold_ids_set = set(other_fold_dataset.get_ids())
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = Dataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(solubility_dataset)
    assert sorted(merged_dataset.get_ids()) == (
           sorted(solubility_dataset.get_ids()))

  def test_singletask_index_k_fold_split(self):
    """
    Test singletask IndexSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    index_splitter = IndexSplitter()
    ids_set = set(solubility_dataset.get_ids())

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = index_splitter.k_fold_split(solubility_dataset, fold_dirs)

    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      # Verify lengths is 10/k == 2
      assert len(fold_dataset) == 2
      # Verify that compounds in this fold are subset of original compounds
      fold_ids_set = set(fold_dataset.get_ids())
      assert fold_ids_set.issubset(ids_set)
      # Verify that no two folds have overlapping compounds.
      for other_fold in range(K):
        if fold == other_fold:
          continue
        other_fold_dataset = fold_datasets[other_fold]
        other_fold_ids_set = set(other_fold_dataset.get_ids())
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = Dataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(solubility_dataset)
    assert sorted(merged_dataset.get_ids()) == (
           sorted(solubility_dataset.get_ids()))
    
  def test_singletask_scaffold_k_fold_split(self):
    """
    Test singletask ScaffoldSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    scaffold_splitter = ScaffoldSplitter()
    ids_set = set(solubility_dataset.get_ids())

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = scaffold_splitter.k_fold_split(
        solubility_dataset, fold_dirs)

    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      # Verify lengths is 10/k == 2
      assert len(fold_dataset) == 2
      # Verify that compounds in this fold are subset of original compounds
      fold_ids_set = set(fold_dataset.get_ids())
      assert fold_ids_set.issubset(ids_set)
      # Verify that no two folds have overlapping compounds.
      for other_fold in range(K):
        if fold == other_fold:
          continue
        other_fold_dataset = fold_datasets[other_fold]
        other_fold_ids_set = set(other_fold_dataset.get_ids())
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = Dataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(solubility_dataset)
    assert sorted(merged_dataset.get_ids()) == (
           sorted(solubility_dataset.get_ids()))

  def test_multitask_random_split(self):
    """
    Test multitask RandomSplitter class.
    """
    multitask_dataset = self.load_multitask_data()
    random_splitter = RandomSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            multitask_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_multitask_index_split(self):
    """
    Test multitask IndexSplitter class.
    """
    multitask_dataset = self.load_multitask_data()
    index_splitter = IndexSplitter()
    train_data, valid_data, test_data = \
        index_splitter.train_valid_test_split(
            multitask_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_multitask_scaffold_split(self):
    """
    Test multitask ScaffoldSplitter class.
    """
    multitask_dataset = self.load_multitask_data()
    scaffold_splitter = ScaffoldSplitter()
    train_data, valid_data, test_data = \
        scaffold_splitter.train_valid_test_split(
            multitask_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_stratified_multitask_split(self):
    """
    Test multitask StratifiedSplitter class
    """
    # sparsity is determined by number of w weights that are 0 for a given
    # task structure of w np array is such that each row corresponds to a
    # sample. The loaded sparse dataset has many rows with only zeros
    sparse_dataset = self.load_sparse_multitask_dataset()
    X, y, w, ids = sparse_dataset.to_numpy()
    
    stratified_splitter = StratifiedSplitter()
    datasets = stratified_splitter.train_valid_test_split(
        sparse_dataset,
        self.train_dir, self.valid_dir, self.test_dir,
        frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    train_data, valid_data, test_data = datasets

    for dataset_index, dataset in enumerate(datasets):
      X, y, w, ids = dataset.to_numpy()
      # verify that there are no rows (samples) in weights matrix w
      # that have no hits.
      assert len(np.where(~w.any(axis=1))[0]) == 0
