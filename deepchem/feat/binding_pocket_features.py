"""
Featurizes proposed binding pockets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "GPL"

import numpy as np
import os
import pybel
import tempfile
import mdtraj as md
from scipy.spatial import ConvexHull
from deepchem.feat import hydrogenate_and_compute_partial_charges
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.grid_featurizer import load_molecule
from deepchem.feat import Featurizer

class BindingPocketFeaturizer(Featurizer):
  """
  Featurizes binding pockets with information about chemical environments.
  """

  residues = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
              "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "PYL", "SER", "SEC",
              "THR", "TRP", "TYR", "VAL", "ASX", "GLX"]

  def featurize(self, protein_file, pockets, pocket_atoms, pocket_coords):
    """
    Calculate atomic coodinates.
    """
    protein = md.load(protein_file)
    n_pockets = len(pockets)
    n_residues = len(BindingPocketFeaturizer.residues)
    res_map = dict(zip(BindingPocketFeaturizer.residues, range(n_residues)))
    all_features = np.zeros((n_pockets, n_residues)) 
    for pocket_num, (pocket, coords) in enumerate(zip(pockets, pocket_coords)):
      atoms = pocket_atoms[pocket]
      for atom in atoms:
        atom_name = str(protein.top.atom(atom))
        # atom_name is of format RESX-ATOMTYPE
        # where X is a 1 to 4 digit number
        residue = atom_name[:3]
        atomtype = atom_name.split("-")[1]
        all_features[pocket_nu, res_map[residue]] += 1
    return all_features 
