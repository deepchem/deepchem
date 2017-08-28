"""
Featurizes proposed binding pockets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import numpy as np
from deepchem.utils.save import log
from deepchem.feat import Featurizer


class BindingPocketFeaturizer(Featurizer):
  """
  Featurizes binding pockets with information about chemical environments.
  """

  residues = [
      "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "PYL", "SER", "SEC", "THR", "TRP",
      "TYR", "VAL", "ASX", "GLX"
  ]

  n_features = len(residues)

  def featurize(self,
                protein_file,
                pockets,
                pocket_atoms_map,
                pocket_coords,
                verbose=False):
    """
    Calculate atomic coodinates.
    """
    import mdtraj
    protein = mdtraj.load(protein_file)
    n_pockets = len(pockets)
    n_residues = len(BindingPocketFeaturizer.residues)
    res_map = dict(zip(BindingPocketFeaturizer.residues, range(n_residues)))
    all_features = np.zeros((n_pockets, n_residues))
    for pocket_num, (pocket, coords) in enumerate(zip(pockets, pocket_coords)):
      pocket_atoms = pocket_atoms_map[pocket]
      for ind, atom in enumerate(pocket_atoms):
        atom_name = str(protein.top.atom(atom))
        # atom_name is of format RESX-ATOMTYPE
        # where X is a 1 to 4 digit number
        residue = atom_name[:3]
        if residue not in res_map:
          log("Warning: Non-standard residue in PDB file", verbose)
          continue
        atomtype = atom_name.split("-")[1]
        all_features[pocket_num, res_map[residue]] += 1
    return all_features
