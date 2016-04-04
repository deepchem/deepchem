"""
Tests for splitter objects. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import unittest
from deepchem.splits import RandomSplitter
from deepchem.splits import ScaffoldSplitter
from deepchem.splits.tests import TestSplitAPI

class TestSplitters(TestSplitAPI):
  """
  Test some basic splitters.
  """
  def test_singletask_random_split(self):
    """Test singletask RandomSplitter class."""
    solubility_samples = self._load_solubility_samples()
    random_splitter = RandomSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            solubility_samples,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1, reload=False)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_singletask_scaffold_split(self):
    """Test singletask ScaffoldSplitter class."""
    solubility_samples = self._load_solubility_samples()
    scaffold_splitter = ScaffoldSplitter()
    train_data, valid_data, test_data = \
        scaffold_splitter.train_valid_test_split(
            solubility_samples,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1, reload=False)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_multitask_random_split(self):  
    """Test multitask RandomSplitter class."""
    multitask_samples = self._load_multitask_samples()
    random_splitter = RandomSplitter()
    train_data, valid_data, test_data = \
        random_splitter.train_valid_test_split(
            multitask_samples,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1, reload=False)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1

  def test_multitask_scaffold_split(self):  
    """Test multitask ScaffoldSplitter class."""
    multitask_samples = self._load_multitask_samples()
    scaffold_splitter = ScaffoldSplitter()
    train_data, valid_data, test_data = \
        scaffold_splitter.train_valid_test_split(
            multitask_samples,
            self.train_dir, self.valid_dir, self.test_dir,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1, reload=False)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1
