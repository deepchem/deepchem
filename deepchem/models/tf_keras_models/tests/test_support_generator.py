"""
Simple Tests for Support Generation 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import numpy as np
import unittest
import tensorflow as tf
from deepchem.datasets import NumpyDataset
from deepchem.models.tf_keras_models.support_classifier import SupportGenerator 

class TestSupportGenerator(unittest.TestCase):
  """
  Test that support generation happens properly.
  """

  def test_simple_support_generator(self):
    """Conducts simple test that support generator runs."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_pos = 1
    n_neg = 5 
    n_trials = 10
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    ############################################## DEBUG
    print("y")
    print(y)
    ############################################## DEBUG
    w = np.ones((n_samples, n_tasks))
    dataset = NumpyDataset(X, y, w, ids)

    # Create support generator
    supp_gen = SupportGenerator(
        dataset, np.arange(n_tasks), n_pos, n_neg, n_trials)

  def test_support_generator_correct_samples(self):
    """Tests that samples from support generator have desired shape."""
    n_samples = 20
    n_features = 3
    n_tasks = 1
    n_pos = 1
    n_neg = 5 
    n_trials = 10
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = NumpyDataset(X, y, w, ids)

    # Create support generator
    supp_gen = SupportGenerator(
        dataset, np.arange(n_tasks), n_pos, n_neg, n_trials)
    num_supports = 0
    
    for (task, support) in supp_gen:
      assert support.X.shape == (n_pos + n_neg, n_features)
      num_supports += 1
      assert task == 0 # Only one task in this example
      n_supp_pos = np.count_nonzero(support.y)
      assert n_supp_pos == n_pos
    assert num_supports == n_trials
