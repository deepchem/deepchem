"""
Docks protein-ligand pairs 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import numpy as np
import os
import tempfile
from deepchem.feat import GridFeaturizer
from deepchem.data import DiskDataset
from deepchem.models import SklearnModel
from deepchem.dock.pose_scoring import PoseScorer
from deepchem.dock.pose_generation import VinaPoseGenerator
from sklearn.ensemble import RandomForestRegressor
from subprocess import call

class Docker(object):
  """Abstract Class specifying API for Docking."""

  def dock(self, protein_file, ligand_file):
    raise NotImplementedError

class VinaGridRFDocker(object):
  """Vina pose-generation, RF-models on grid-featurization of complexes."""

  def __init__(self, subset="refined", n_trees=100):
    """Builds model."""
    self.base_dir = tempfile.mkdtemp()
    call(("wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/%s_grid.tar.gz" % subset).split())
    call(("tar -zxvf %s_grid.tar.gz" % subset).split())
    call(("mv %s_grid %s" % (subset, self.base_dir)).split())
    refined_dir = os.path.join(self.base_dir, "%s_grid" % subset)
    self.dataset = DiskDataset(refined_dir)

    # Fit model on dataset
    sklearn_model = RandomForestRegressor(n_estimators=n_trees)
    model = SklearnModel(sklearn_model)
    print("About to fit model on refined set")
    model.fit(self.dataset)

    self.pose_scorer = PoseScorer(model, feat="grid")
    self.pose_generator = VinaPoseGenerator() 

  def dock(self, protein_file, ligand_file):
    """Docks using Vina and RF."""
    protein_docked, ligand_docked = self.pose_generator.generate_poses(
        protein_file, ligand_file)
    score = self.pose_scorer.score(protein_docked, ligand_docked)
    return (score, (protein_docked, ligand_docked))


