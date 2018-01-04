import deepchem as dc
import numpy as np
import unittest


class TestDNASim(unittest.TestCase):

  def test_motif_density_localization_simulation(self):
    "Test motif density localization simulation." ""
    motif_density_localization_simulation_parameters = {
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
        **motif_density_localization_simulation_parameters)
    assert sequences.shape == (60,)
    assert y.shape == (60, 1)

  def test_motif_counting_simulation(self):
    "Test motif counting"
    motif_count_simulation_parameters = {
        "motif_name": "TAL1_known4",
        "seq_length": 1000,
        "pos_counts": [5, 10],
        "neg_counts": [1, 2],
        "num_pos": 30,
        "num_neg": 30,
        "GC_fraction": 0.4
    }
    sequences, y, embed = dc.molnet.simulate_motif_counting(
        **motif_count_simulation_parameters)
    assert sequences.shape == (60,)
    assert y.shape == (60, 1)
