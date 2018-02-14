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
