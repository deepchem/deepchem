"""
Scores protein-ligand poses using DeepChem.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deepchem.feat import RdkitGridFeaturizer

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import os
import tempfile
from deepchem.data import NumpyDataset
from subprocess import call


class PoseScorer(object):
  """Abstract superclass for all scoring methods."""

  def score(self, protein_file, ligand_file):
    """Returns a score for a protein/ligand pair."""
    raise NotImplementedError


class GridPoseScorer(object):

  def __init__(self, model, feat="grid"):
    """Initializes a pose-scorer."""
    self.model = model
    if feat == "grid":
      self.featurizer = RdkitGridFeaturizer(
          voxel_width=16.0,
          feature_types="voxel_combined",
          # TODO(rbharath, enf): Figure out why pi_stack is slow and cation_pi
          # causes segfaults.
          #voxel_feature_types=["ecfp", "splif", "hbond", "pi_stack", "cation_pi",
          #"salt_bridge"], ecfp_power=9, splif_power=9,
          voxel_feature_types=["ecfp", "splif", "hbond", "salt_bridge"],
          ecfp_power=9,
          splif_power=9,
          parallel=True,
          flatten=True)
    else:
      raise ValueError("feat not defined.")

  def score(self, protein_file, ligand_file):
    """Returns a score for a protein/ligand pair."""
    features = self.featurizer.featurize_complexes([ligand_file],
                                                   [protein_file])
    dataset = NumpyDataset(X=features, y=None, w=None, ids=None)
    score = self.model.predict(dataset)
    return score
