"""
Simple Tests for Support Generation 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

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
    n_samples = 10
    n_features = 3
    n_tasks = 1
    n_pos = 1
    n_neg = 10
    n_trials = 10
    
    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_tasks)
    w = np.ones((n_samples, n_tasks))
    dataset = NumpyDataset(X, y, w, ids)

    # Create support generator
    supp_gen = SupportGenerator(
        dataset, np.arange(n_tasks), n_pos, n_neg, n_trials)
  

