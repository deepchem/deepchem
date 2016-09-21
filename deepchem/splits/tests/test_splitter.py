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

  def test_singletask_random_k_fold_split(self):
    """
    Test singletask RandomSplitter class.
    """
    solubility_dataset = self.load_solubility_data()
    random_splitter = RandomSplitter()
    ids_set = set(solubility_dataset.get_ids())
    #################################################### DEBUG
    print("ids_set")
    print(ids_set)
    #################################################### DEBUG

    K = 5
    fold_dirs = [tempfile.mkdtemp() for i in range(K)]
    fold_datasets = random_splitter.k_fold_split(solubility_dataset, fold_dirs)
    #################################################### DEBUG
    for fold in range(K):
      fold_dataset = fold_datasets[fold]
      fold_ids_set = set(fold_dataset.get_ids())
      print("fold")
      print(fold)
      print("fold_ids_set")
      print(fold_ids_set)
    #################################################### DEBUG
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
        #################################################### DEBUG
        print("fold, other_fold")
        print(fold, other_fold)
        print("fold_ids_set, other_fold_ids_set")
        print(fold_ids_set, other_fold_ids_set)
        #################################################### DEBUG
        assert fold_ids_set.isdisjoint(other_fold_ids_set)

    merge_dir = tempfile.mkdtemp()
    merged_dataset = Dataset.merge(merge_dir, fold_datasets)
    assert len(merged_dataset) == len(solubility_dataset)
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
    # ensure sparse dataset is actually sparse

    sparse_dataset = self.load_sparse_multitask_dataset()

    X, y, w, ids = sparse_dataset.to_numpy()

    
    # sparsity is determined by number of w weights that are 0 for a given
    # task structure of w np array is such that each row corresponds to a
    # sample -- e.g., analyze third column for third sparse task
  
    frac_train = 0.5
    cutoff = int(frac_train * w.shape[0])
    w = w[:cutoff, :]
    sparse_flag = False

    col_index = 0
    for col in w.T:
      if not np.any(col): #check to see if any columns are all zero
        sparse_flag = True
        break
      col_index+=1
    if not sparse_flag:
      print("Test dataset isn't sparse -- test failed")
    else:
      print("Column %d is sparse -- expected" % col_index)
    assert sparse_flag

    stratified_splitter = StratifiedSplitter()
    train_data, valid_data, test_data = \
        stratified_splitter.train_valid_test_split(
            sparse_dataset,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1
        )

    datasets = [train_data, valid_data, test_data]
    dataset_index = 0
    for dataset in datasets:
      X, y, w, ids = dataset.to_numpy()
      # verify that each task in the train dataset has some hits
      for col in w.T:
          if not np.any(col):
              print("Fail -- one column doesn't have results")
              if dataset_index == 0:
                  print("train_data failed")
              elif dataset_index == 1:
                  print("valid_data failed")
              elif dataset_index == 2:
                  print("test_data failed")
              assert np.any(col)
      if dataset_index == 0:
          print("train_data passed")
      elif dataset_index == 1:
          print("valid_data passed")
      elif dataset_index == 2:
          print("test_data passed")
      dataset_index+=1
    print("end of stratified test")
    assert 1 == 1
