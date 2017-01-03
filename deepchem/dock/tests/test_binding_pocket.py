"""
Tests for Pose Generation 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import unittest
import tempfile
import os
import shutil
import numpy as np
import deepchem as dc

class TestPoseGeneration(unittest.TestCase):
  """
  Does sanity checks on pose generation. 
  """

  def test_convex_rf_init(self):
    """Tests that ConvexHullRFPocketFinder can be initialized."""
    finder = dc.dock.ConvexHullRFPocketFinder()

  def test_convex_rf_find_all_pockets(self):
    """Tests that binding pockets are detected."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")

    finder = dc.dock.ConvexHullRFPocketFinder()

    all_pockets = finder.find_all_pockets(protein_file)
    assert isinstance(all_pockets, list)
    # Pocket is of form [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    for pocket in all_pockets:
      assert len(pocket) == 3
      assert len(pocket[0]) == 2
      assert len(pocket[1]) == 2
      assert len(pocket[2]) == 2

  def test_convex_rf_find_pockets(self):
    """Test that some pockets are filtered out."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")

    finder = dc.dock.ConvexHullRFPocketFinder()

    all_pockets = finder.find_all_pockets(protein_file)
    pockets = finder.find_pockets(protein_file)

    assert len(pockets) < len(all_pockets)
