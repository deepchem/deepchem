"""
Tests for splitter objects.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

from deepchem.splits import RandomSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import StratifiedSplitter
from deepchem.datasets.tests import TestDatasetAPI
import numpy as np


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

      """
      sparsity is determined by number of w weights that are 0 for a given task
      structure of w np array is such that each row corresponds to a sample -- e.g., analyze third column for third
      sparse task
      """
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
