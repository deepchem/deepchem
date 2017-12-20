"""
Testing construction of graph models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import tensorflow as tf
import deepchem as dc
from tensorflow.python.framework import test_util


class TestGraphModels(test_util.TensorFlowTestCase):
  """
  Test Container usage.
  """

  def setUp(self):
    super(TestGraphModels, self).setUp()
    self.root = '/tmp'
