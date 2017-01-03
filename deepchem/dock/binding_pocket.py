"""
Generates protein-ligand docked poses using Autodock Vina.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import numpy as np
import os
import pybel
import tempfile
from scipy.spatial import ConvexHull
from deepchem.feat import hydrogenate_and_compute_partial_charges
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.grid_featurizer import load_molecule
from subprocess import call

class BindingPocketFinder(object):
  """Abstract superclass for binding pocket detection"""

  def find_pockets(self, protein_file):
    """Finds potential binding pockets in proteins."""
    raise NotImplementedError

class ConvexHullRFPocketFinder(BindingPocketFinder):
  """Implementation that uses convex hull of protein to find pockets.

  Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4112621/pdf/1472-6807-14-18.pdf
  """
  def __init__(self):
    pass

  def find_all_pockets(self, protein_file):
    """Find list of binding pockets on protein."""
    # protein_coords is (N, 3) tensor
    coords = load_molecule(protein_file)[0]
    hull = ConvexHull(coords)
    faces = []
    for triangle in hull.simplices:
      # coords[triangle, 0] gives the x-dimension of all triangle points
      # Take transpose to make sure rows correspond to atoms.
      points = np.array(
          [coords[triangle, 0], coords[triangle, 1], coords[triangle, 2]]).T
      x_min, x_max = np.amin(points[:, 0]), np.amax(points[:, 0])
      y_min, y_max = np.amin(points[:, 1]), np.amax(points[:, 1])
      z_min, z_max = np.amin(points[:, 2]), np.amax(points[:, 2])
      faces.append([(x_min, x_max), (y_min, y_max), (z_min, z_max)])
    return faces

  def find_pockets(self, protein_file):
    """Find list of suitable binding pockets on protein."""
    return self.find_all_pockets(protein_file)
