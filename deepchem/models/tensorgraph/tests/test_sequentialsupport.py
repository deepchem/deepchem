import unittest
import numpy as np
import deepchem as dc
from deepchem.models.tensorgraph.layers import Dense
from deepchem.models.tensorgraph.layers import SoftMax
from nose.tools import assert_true


class TestSequentialSupport(unittest.TestCase):
  """
  Test that sequential support graphs work correctly.
  """

  def test_initialization(self):
    seq_support = dc.models.SequentialSupport()
