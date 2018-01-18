"""
Tests that FASTA files can be loaded.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__license__ = "MIT"

import os
import unittest
import deepchem as dc


class TestFASTALoader(unittest.TestCase):
  """
  Test FASTALoader 
  """

  def setUp(self):
    super(TestFASTALoader, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_fasta_load(self):
    input_file = os.path.join(self.current_dir,
                              "../../data/tests/example.fasta")
    loader = dc.data.FASTALoader()
    sequences = loader.featurize(input_file)
    # example.fasta contains 3 sequences each of length 58.
    # The one-hot encoding turns base-pairs into vectors of length 4.
    # There is one "image channel")
    assert sequences.X.shape == (3, 4, 58, 1)
