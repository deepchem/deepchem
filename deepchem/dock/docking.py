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
      complexes = outputs
    # We know use_pose_generator_scores == False in this case
    if self.scoring_model is not None:
      for posed_complex in complexes:
        # TODO: How to handle the failure here?
        features, _ = self.featurizer.featurize([molecular_complex])
        dataset = NumpyDataset(X=features)
        score = self.scoring_model.predict(dataset)
        yield (posed_complex, score)
    elif use_pose_generator_scores:
      for posed_complex, score in zip(complexes, scores):
        yield (posed_complex, score)
    else:
      for posed_complex in complexes:
        yield posed_complex
      score = np.zeros((1,))
    return (score, (protein_docked, ligand_docked))
