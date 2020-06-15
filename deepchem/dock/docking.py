"""
Docks Molecular Complexes 
"""
import logging
import numpy as np
import os
import tempfile
from subprocess import call
from deepchem.data import NumpyDataset

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

  def __init__(self, pose_generator, featurizer=None, scoring_model=None):
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
    if ((featurizer is not None and scoring_model is None) or
        (featurizer is None and scoring_model is not None)):
      raise ValueError(
          "featurizer/scoring_model must both be set or must both be None.")
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
           out_dir=None,
           use_pose_generator_scores=False):
    """Generic docking function.

    This docking function uses this object's featurizer, pose
    generator, and scoring model to make docking predictions. This
    function is written in generic style so  

    Parameters
    ----------
    molecular_complex: Object
      Some representation of a molecular complex.
    exhaustiveness: int, optional (default 10)
      Tells pose generator how exhaustive it should be with pose
      generation.
    num_modes: int, optional (default 9)
      Tells pose generator how many binding modes it should generate at
      each invocation.
    num_pockets: int, optional (default None)
      If specified, `self.pocket_finder` must be set. Will only
      generate poses for the first `num_pockets` returned by
      `self.pocket_finder`.
    out_dir: str, optional (default None)
      If specified, write generated poses to this directory.
    use_pose_generator_scores: bool, optional (default False)
      If `True`, ask pose generator to generate scores. This cannot be
      `True` if `self.featurizer` and `self.scoring_model` are set
      since those will be used to generate scores in that case. 

    Returns
    -------
    A generator. If `use_pose_generator_scores==True` or
    `self.scoring_model` is set, then will yield tuples
    `(posed_complex, score)`. Else will yield `posed_complex`.
    """
    if self.scoring_model is not None and use_pose_generator_scores:
      raise ValueError(
          "Cannot set use_pose_generator_scores=True when self.scoring_model is set (since both generator scores for complexes)."
      )
    outputs = self.pose_generator.generate_poses(
        molecular_complex,
        centroid=centroid,
        box_dims=box_dims,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        num_pockets=num_pockets,
        out_dir=out_dir,
        generate_scores=use_pose_generator_scores)
    if use_pose_generator_scores:
      complexes, scores = outputs
    else:
      complexes = outputs
    # We know use_pose_generator_scores == False in this case
    if self.scoring_model is not None:
      for posed_complex in complexes:
        # TODO: How to handle the failure here?
        features, _ = self.featurizer.featurize_complexes([molecular_complex])
        dataset = NumpyDataset(X=features)
        score = self.scoring_model.predict(dataset)
        yield (posed_complex, score)
    elif use_pose_generator_scores:
      for posed_complex, score in zip(complexes, scores):
        yield (posed_complex, score)
    else:
      for posed_complex in complexes:
        yield posed_complex
