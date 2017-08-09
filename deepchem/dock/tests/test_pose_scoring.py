"""
Tests for Pose Scoring 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import sys
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

  def setUp(self):
    """Downloads dataset."""
    call(
        "wget -nv -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/core_grid.tar.gz".
        split())
    call("tar -zxvf core_grid.tar.gz".split())
    self.core_dataset = dc.data.DiskDataset("core_grid/")

  def tearDown(self):
    """Removes dataset"""
    call("rm -rf core_grid/".split())

  def test_pose_scorer_init(self):
    """Tests that pose-score works."""
    if sys.version_info >= (3, 0):
      return
    sklearn_model = RandomForestRegressor(n_estimators=10)
    model = dc.models.SklearnModel(sklearn_model)
    print("About to fit model on core set")
    model.fit(self.core_dataset)

    pose_scorer = dc.dock.GridPoseScorer(model, feat="grid")

  def test_pose_scorer_score(self):
    """Tests that scores are generated"""
    if sys.version_info >= (3, 0):
      return
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    sklearn_model = RandomForestRegressor(n_estimators=10)
    model = dc.models.SklearnModel(sklearn_model)
    print("About to fit model on core set")
    model.fit(self.core_dataset)

    pose_scorer = dc.dock.GridPoseScorer(model, feat="grid")
    score = pose_scorer.score(protein_file, ligand_file)
    assert score.shape == (1,)
