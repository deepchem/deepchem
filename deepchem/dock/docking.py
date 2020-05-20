"""
Docks Molecular Complexes 
"""
import logging
import numpy as np
import os
import tempfile
from subprocess import call

logger = logging.getLogger(__name__)


class Docker(object):
  """A generic molecular docking class

  This class provides a docking engine which uses provided models for
  featurization, pose generation, and scoring. Most pieces of docking
  software are command line tools that are invoked from the shell. The
  goal of this class is to provide a python clean API for invoking
  molecular docking programmatically.

  The implementation of this class is lightweight and generic. It's
  expected that the majority of the heavy lifting will be done by pose
  generation and scoring classes that are provided to this class.
  """

  def __init__(self,
               pose_generator,
               featurizer=None,
               scoring_model=None):
    """Builds model.

    Parameters
    ----------
    pose_generator: `PoseGenerator`
      The pose generator to use for this model
    featurizer: `ComplexFeaturizer`
      Featurizer associated with `scoring_model`
    scoring_model: `Model`
      Should make predictions on molecular complex.
    """
    self.base_dir = tempfile.mkdtemp()
    self.pose_generator = pose_generator
    self.featurizer = featurizer
    self.scoring_model = scoring_model

  def dock(self,
           molecular_complex,
           centroid=None,
           box_dims=None,
           exhaustiveness=10,
           num_modes=9,
           num_pockets=None,
           out_dir=None):
    """Docks using Vina and RF.

    Parameters
    ----------
    molecular_complex: Object
      Some representation of a molecular complex.
    exhaustiveness: int, optional (default 10)
      Tells Autodock Vina how exhaustive it should be with pose
      generation.
    num_modes: int, optional (default 9)
      Tells Autodock Vina how many binding modes it should generate at
      each invocation.
    num_pockets: int, optional (default None)
      If specified, `self.pocket_finder` must be set. Will only
      generate poses for the first `num_pockets` returned by
      `self.pocket_finder`.
    out_dir: str, optional
      If specified, write generated poses to this directory.
    """
    complexes = self.pose_generator.generate_poses(molecular_complex,
                                                   centroid=centroid,
                                                   box_dims=box_dims,
                                                   exhaustiveness=exhaustiveness,
                                                   num_modes=num_modes,
                                                   num_pockets=num_pockets,
                                                   out_dir=out_dir)
    for posed_complex in complexes:
      if self.featurizer is not None:
        # TODO: How to handle the failure here?
        features, _ = self.featurizer.featurize_complexes([molecular_complex])
        dataset = NumpyDataset(X=features)
        score = self.model.predict(dataset)
        yield (score, posed_complex)
      else:
        yield posed_complex
