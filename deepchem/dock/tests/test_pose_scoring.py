"""
Tests for Pose Scoring
"""

import logging
import unittest
import numpy as np

from deepchem.dock.pose_scoring import vina_nonlinearity
from deepchem.dock.pose_scoring import vina_hydrophobic
from deepchem.dock.pose_scoring import vina_gaussian_first
from deepchem.dock.pose_scoring import vina_gaussian_second
from deepchem.dock.pose_scoring import vina_hbond
from deepchem.dock.pose_scoring import vina_repulsion
from deepchem.dock.pose_scoring import cutoff_filter
from deepchem.dock.pose_scoring import vina_energy_term

logger = logging.getLogger(__name__)


class TestPoseScoring(unittest.TestCase):
    """Does sanity checks on pose generation."""

    def test_cutoff_filter(self):
        N = 10
        M = 5
        d = np.ones((N, M))
        x = np.random.rand(N, M)
        cutoff_dist = 0.5
        x_thres = cutoff_filter(d, x, cutoff=cutoff_dist)
        assert (x_thres == np.zeros((N, M))).all()

    def test_vina_nonlinearity(self):
        N = 10
        M = 5
        c = np.random.rand(N, M)
        Nrot = 5
        w = 0.5
        out_tensor = vina_nonlinearity(c, w, Nrot)
        assert out_tensor.shape == (N, M)
        assert (out_tensor == c / (1 + w * Nrot)).all()

    def test_vina_repulsion(self):
        N = 10
        M = 5
        d = np.ones((N, M))
        out_tensor = vina_repulsion(d)
        assert out_tensor.shape == (N, M)
        # Where d is greater than zero, the repulsion is just zeros
        assert (out_tensor == np.zeros_like(d)).all()

    def test_vina_hydrophobic(self):
        N = 10
        M = 5
        d = np.zeros((N, M))
        out_tensor = vina_hydrophobic(d)
        assert out_tensor.shape == (N, M)
        # When d is 0, this should just be 1
        assert (out_tensor == np.ones_like(d)).all()

    def test_vina_hbond(self):
        N = 10
        M = 5
        d = np.zeros((N, M))
        out_tensor = vina_hbond(d)
        assert out_tensor.shape == (N, M)
        # When d == 0, the hbond interaction is 0
        assert (out_tensor == np.zeros_like(d)).all()

    def test_vina_gaussian(self):
        N = 10
        M = 5
        d = np.zeros((N, M))
        out_tensor = vina_gaussian_first(d)
        assert out_tensor.shape == (N, M)
        # The exponential returns 1 when input 0.
        assert (out_tensor == np.ones_like(d)).all()

        d = 3 * np.ones((N, M))
        out_tensor = vina_gaussian_second(d)
        assert out_tensor.shape == (N, M)
        # This exponential returns 1 when input 3
        assert (out_tensor == np.ones_like(d)).all()

    def test_energy_term(self):
        N = 10
        M = 5
        coords1 = np.random.rand(N, 3)
        coords2 = np.random.rand(M, 3)
        weights = np.ones((5,))
        wrot = 1.0
        Nrot = 3
        energy = vina_energy_term(coords1, coords2, weights, wrot, Nrot)
        assert energy > 0
