"""
Test that Layers work as advertised.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import unittest
import deepchem as dc
from tensorflow.python.framework import test_util


class TestLayers(test_util.TensorFlowTestCase):
  """
  Test Layers.

  The tests in this class only do basic sanity checks to make sure that
  produced tensors have the right shape.
  """

  def setUp(self):
    super(TestLayers, self).setUp()
    self.root = '/tmp'

  def test_dense(self):
    """Tests dense layer class can be initialized."""
    with self.test_session() as sess:
      dense = dc.nn.Dense(32, input_dim=16)

  def test_dropout(self):
    """Tests that dropout can be initialized."""
    with self.test_session() as sess:
      dropout = dc.nn.Dropout(.5)

  def test_input(self):
    """Tests that inputs can be created."""
    with self.test_session() as sess:
      input_layer = dc.nn.Input(shape=(32,))
