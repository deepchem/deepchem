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

  def test_vina_poses(self):
    """Test that VinaPoseGenerator Functions."""
    vpg = dc.dock.VinaPoseGenerator()

