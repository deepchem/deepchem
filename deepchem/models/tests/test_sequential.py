"""
Integration tests for singletask vector feature models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import unittest
import tempfile
import shutil
import tensorflow as tf
import numpy as np
import deepchem as dc


class TestSequential(unittest.TestCase):
  """
  Test API for sequential models 
  """

  def test_model_construct(self):
    """Test that models can be constructed with Sequential."""
    model = dc.models.Sequential()
    model.add_features(dc.nn.Input(shape=(32,)))
    model.add_labels(dc.nn.Input(shape=(1,)))
    model.add(dc.nn.Dense(32, 32))
    model.add(dc.nn.Dense(32, 32))

  def test_model_fit(self):
    """Test that models can be fit"""
    model = dc.models.Sequential()
    model.add_features(dc.nn.Input(shape=(32,)))
    model.add_labels(dc.nn.Input(shape=(1,)))
    model.add(dc.nn.Dense(32, 32))
    model.add(dc.nn.Dense(1, 32))
    model.add_loss(dc.nn.mean_squared_error)

    X = np.zeros((10, 32))
    y = np.zeros((10,))
    dataset = dc.data.NumpyDataset(X, y)
    model.fit(dataset)
