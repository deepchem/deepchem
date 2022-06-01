"""
Tests that sequence handling utilities work.
"""
__author__ = "Bharath Ramsundar"
__license__ = "MIT"

import unittest
import os

import numpy as np

import deepchem as dc

LETTERS = "XYZ"


class TestSeq(unittest.TestCase):
  """
  Tests sequence handling utilities.
  """

  def setUp(self):
    super(TestSeq, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))

  def test_one_hot_simple(self):
    sequences = np.array(["ACGT", "GATA", "CGCG"])
    sequences = dc.utils.genomics_utils.seq_one_hot_encode(sequences)
    self.assertEqual(sequences.shape, (3, 5, 4, 1))

  def test_one_hot_mismatch(self):
    # One sequence has length longer than others. This should throw a
    # ValueError.

    with self.assertRaises(ValueError):
      sequences = np.array(["ACGTA", "GATA", "CGCG"])
      sequences = dc.utils.genomics_utils.seq_one_hot_encode(sequences)

  def test_encode_fasta_sequence(self):
    # Test it's possible to load a sequence with an aribrary alphabet from a fasta file.
    fname = os.path.join(self.current_dir, "./assets/example.fasta")

    encoded_seqs = dc.utils.genomics_utils.encode_bio_sequence(fname,
                                                               letters=LETTERS)
    expected = np.expand_dims(
        np.array([
            [[1, 0], [0, 1], [0, 0]],
            [[0, 1], [0, 0], [1, 0]],
        ]), -1)

    np.testing.assert_array_equal(expected, encoded_seqs)

  def test_encode_fastq_sequence(self):
    fname = os.path.join(self.current_dir, "./assets/example.fastq")

    encoded_seqs = dc.utils.genomics_utils.encode_bio_sequence(
        fname, file_type="fastq", letters=LETTERS)

    expected = np.expand_dims(
        np.array([
            [[1, 0], [0, 1], [0, 0]],
            [[0, 1], [0, 0], [1, 0]],
        ]), -1)

    np.testing.assert_array_equal(expected, encoded_seqs)
