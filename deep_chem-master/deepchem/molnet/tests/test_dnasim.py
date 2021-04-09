import deepchem as dc
import numpy as np
import unittest


class TestDNASim(unittest.TestCase):

  def test_motif_density_localization_simulation(self):
    "Test motif density localization simulation." ""
    params = {
        "motif_name": "TAL1_known4",
        "seq_length": 1000,
        "center_size": 150,
        "min_motif_counts": 2,
        "max_motif_counts": 4,
        "num_pos": 30,
        "num_neg": 30,
        "GC_fraction": 0.4
    }
    sequences, y, embed = dc.molnet.simulate_motif_density_localization(
        **params)
    assert sequences.shape == (60,)
    assert y.shape == (60, 1)

  def test_motif_counting_simulation(self):
    "Test motif counting"
    params = {
        "motif_name": "TAL1_known4",
        "seq_length": 1000,
        "pos_counts": [5, 10],
        "neg_counts": [1, 2],
        "num_pos": 30,
        "num_neg": 30,
        "GC_fraction": 0.4
    }
    sequences, y, embed = dc.molnet.simulate_motif_counting(**params)
    assert sequences.shape == (60,)
    assert y.shape == (60, 1)

  def test_simple_motif_embedding(self):
    "Test simple motif embedding"
    params = {
        "motif_name": "TAL1_known4",
        "seq_length": 1000,
        "num_seqs": 30,
        "GC_fraction": 0.4
    }
    sequences, embed = dc.molnet.simple_motif_embedding(**params)
    assert sequences.shape == (30,)

  def test_motif_density(self):
    "Test motif density"
    params = {
        "motif_name": "TAL1_known4",
        "seq_length": 1000,
        "num_seqs": 30,
        "min_counts": 2,
        "max_counts": 4,
        "GC_fraction": 0.4
    }
    sequences, embed = dc.molnet.motif_density(**params)
    assert sequences.shape == (30,)

  def test_single_motif_detection(self):
    "Test single motif detection"
    params = {
        "motif_name": "TAL1_known4",
        "seq_length": 1000,
        "num_pos": 30,
        "num_neg": 30,
        "GC_fraction": 0.4
    }
    sequences, y, embed = dc.molnet.simulate_single_motif_detection(**params)
    assert sequences.shape == (60,)
    assert y.shape == (60, 1)
