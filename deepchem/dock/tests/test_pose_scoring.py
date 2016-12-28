"""
Tests for Pose Scoring 
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
from sklearn.ensemble import RandomForestRegressor
from subprocess import call

class TestPoseScoring(unittest.TestCase):
  """
  Does sanity checks on pose generation. 
  """

  def test_pose_scorer_init(self):
    """Tests that pose-score works."""
    call("wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz".split())
    call("tar -zxvf core_grid.tar.gz".split())
    core_dataset = dc.data.DiskDataset("core_grid/")

    sklearn_model = RandomForestRegressor(n_estimators=10)
    model = dc.models.SklearnModel(sklearn_model)
    print("About to fit model on core set")
    model.fit(core_dataset)

    pose_scorer = dc.dock.PoseScorer(model, feat="grid")
