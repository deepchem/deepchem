"""
Test that genomic metrics work.
"""
from __future__ import division
from __future__ import unicode_literals

import unittest
import os

import numpy as np
import deepchem as dc

LETTERS = "ACGT"

from deepchem.metrics.genomic_metrics import get_motif_scores
from deepchem.metrics.genomic_metrics import get_pssm_scores

class TestGenomicMetrics(unittest.TestCase):
  """
  Tests that genomic metrics work as expected. 
  """

  def test_get_motif_scores(self):
    # Encode motif
    motif_name = "TAL1_known4"
    sequences = np.array(["ACGTA", "GATAG", "CGCGC"])
    sequences = dc.utils.save.seq_one_hot_encode(sequences, letters=LETTERS)
    # sequences now has shape (3, 4, 5, 1)
    self.assertEqual(sequences.shape, (3, 4, 5, 1))

    get_motif_scores(sequences, [motif_name])

  def test_get_pssm_scores(self):
    motif_name = "TAL1_known4"
    sequences = np.array(["ACGTA", "GATAG", "CGCGC"])
    sequences = dc.utils.save.seq_one_hot_encode(sequences, letters=LETTERS)
    # sequences now has shape (3, 4, 5, 1)
    self.assertEqual(sequences.shape, (3, 4, 5, 1))

    get_pssm_scores(sequences, [motif_name])
