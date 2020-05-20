"""
Tests for Pose Scoring
"""
import sys
import logging
import unittest
import tempfile
import os
import shutil
import numpy as np
import pytest

import deepchem as dc
from subprocess import call
from deepchem.dock.pose_scoring import vina_nonlinearity
from deepchem.dock.pose_scoring import vina_hydrophobic
from deepchem.dock.pose_scoring import vina_hbond
from deepchem.dock.pose_scoring import vina_repulsion
from deepchem.dock.pose_scoring import cutoff_filter
from deepchem.dock.pose_scoring import vina_energy_term

logger = logging.getLogger(__name__)


@pytest.mark.linux_only
class TestPoseScoring(unittest.TestCase):
  """
  Does sanity checks on pose generation.
  """

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

  def test_vina_repulsion(self):
    N = 10
    M = 5
    d = np.random.rand(N, M)
    out_tensor = vina_repulsion(d)
    assert out_tensor.shape == (N, M)
    
  def test_vina_hydrophobic(self):
    N = 10
    M = 5
    d = np.random.rand(N, M)
    out_tensor = vina_hydrophobic(d)
    assert out_tensor.shape == (N, M)

  def test_vina_hbond(self):
    N = 10
    M = 5
    d = np.random.rand(N, M)
    out_tensor = vina_hbond(d)
    assert out_tensor.shape == (N, M)

  def test_energy_term(self):
    N = 10
    M = 5
    coords1 = np.random.rand(N, 3)
    coords2 = np.random.rand(M, 3)
    weights = np.ones((5,))
    wrot = 1.0
    Nrot = 3
    energy = vina_energy_term(coords1, coords2, weights, wrot, Nrot)
