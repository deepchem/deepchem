"""
Docks protein-ligand pairs 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import os
import tempfile
from deepchem.data import DiskDataset
from deepchem.models import SklearnModel
from deepchem.models import TensorflowMultiTaskRegressor
from deepchem.dock.pose_scoring import GridPoseScorer
from deepchem.dock.pose_generation import VinaPoseGenerator
from sklearn.ensemble import RandomForestRegressor
from subprocess import call


class Docker(object):
  """Abstract Class specifying API for Docking."""

  def dock(self,
           protein_file,
           ligand_file,
           centroid=None,
           box_dims=None,
           dry_run=False):
    raise NotImplementedError


class VinaGridRFDocker(Docker):
  """Vina pose-generation, RF-models on grid-featurization of complexes."""

  def __init__(self, exhaustiveness=10, detect_pockets=False):
    """Builds model."""
    self.base_dir = tempfile.mkdtemp()
    print("About to download trained model.")
    call((
        "wget -nv -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/trained_models/random_full_RF.tar.gz"
    ).split())
    call(("tar -zxvf random_full_RF.tar.gz").split())
    call(("mv random_full_RF %s" % (self.base_dir)).split())
    self.model_dir = os.path.join(self.base_dir, "random_full_RF")

    # Fit model on dataset
    model = SklearnModel(model_dir=self.model_dir)
    model.reload()

    self.pose_scorer = GridPoseScorer(model, feat="grid")
    self.pose_generator = VinaPoseGenerator(
        exhaustiveness=exhaustiveness, detect_pockets=detect_pockets)

  def dock(self,
           protein_file,
           ligand_file,
           centroid=None,
           box_dims=None,
           dry_run=False):
    """Docks using Vina and RF."""
    protein_docked, ligand_docked = self.pose_generator.generate_poses(
        protein_file, ligand_file, centroid, box_dims, dry_run)
    if not dry_run:
      score = self.pose_scorer.score(protein_docked, ligand_docked)
    else:
      score = np.zeros((1,))
    return (score, (protein_docked, ligand_docked))


class VinaGridDNNDocker(object):
  """Vina pose-generation, DNN-models on grid-featurization of complexes."""

  def __init__(self, exhaustiveness=10, detect_pockets=False):
    """Builds model."""
    self.base_dir = tempfile.mkdtemp()
    print("About to download trained model.")
    call((
        "wget -nv -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/trained_models/random_full_DNN.tar.gz"
    ).split())
    call(("tar -zxvf random_full_DNN.tar.gz").split())
    call(("mv random_full_DNN %s" % (self.base_dir)).split())
    self.model_dir = os.path.join(self.base_dir, "random_full_DNN")

    # Fit model on dataset
    pdbbind_tasks = ["-logKd/Ki"]
    n_features = 2052
    model = TensorflowMultiTaskRegressor(
        len(pdbbind_tasks),
        n_features,
        logdir=self.model_dir,
        dropouts=[.25],
        learning_rate=0.0003,
        weight_init_stddevs=[.1],
        batch_size=64)
    model.reload()

    self.pose_scorer = GridPoseScorer(model, feat="grid")
    self.pose_generator = VinaPoseGenerator(
        exhaustiveness=exhaustiveness, detect_pockets=detect_pockets)

  def dock(self,
           protein_file,
           ligand_file,
           centroid=None,
           box_dims=None,
           dry_run=False):
    """Docks using Vina and DNNs."""
    protein_docked, ligand_docked = self.pose_generator.generate_poses(
        protein_file, ligand_file, centroid, box_dims, dry_run)
    if not dry_run:
      score = self.pose_scorer.score(protein_docked, ligand_docked)
    else:
      score = np.zeros((1,))
    return (score, (protein_docked, ligand_docked))
