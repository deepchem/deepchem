"""
Tests that sequence handling utilities work.
"""
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__license__ = "MIT"

import numpy as np
import unittest
import deepchem as dc


class TestSeq(unittest.TestCase):
  """
  Tests sequence handling utilities.
  """

  def test_one_hot_simple(self):
    sequences = np.array(["ACGT", "GATA", "CGCG"])
    sequences = dc.utils.save.seq_one_hot_encode(sequences)
    assert sequences.shape == (3, 4, 4, 1)

  def test_one_hot_mismatch(self):
    # One sequence has length longer than others. This should throw a
    # value error.
    thrown = False
    try:
      sequences = np.array(["ACGTA", "GATA", "CGCG"])
      sequences = dc.utils.save.seq_one_hot_encode(sequences)
    except ValueError:
      thrown = True
    assert thrown

  def test_encode_sequence_with_biopython(self):
      fname = "./data/example.fasta"

      encoded_seqs = dc.utils.save.encode_sequence_with_biopython(fname)
      expected = np.array([
        [[0, 0], [1, 0], [0, 0], [0, 1], [0, 0]],
        [[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]],
      ])

      np.testing.assert_array_equal(expected, encoded_seqs)
