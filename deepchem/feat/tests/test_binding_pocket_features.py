"""
Test atomic coordinates and neighbor lists.
"""
import os
import numpy as np
import unittest
from rdkit import Chem
import deepchem as dc

class TestAtomicCoordinates(unittest.TestCase):
  """
  Test AtomicCoordinates.
  """

  def test_atomic_coordinates(self):
    """
    Simple test that atomic coordinates returns ndarray of right shape.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "../../dock/tests/1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")

    finder = dc.dock.ConvexHullPocketFinder()
    pocket_featurizer = dc.feat.BindingPocketFeaturizer()
    pockets, pocket_atoms, pocket_coords = finder.find_pockets(protein_file, ligand_file)
    n_pockets = len(pockets)
    
    pocket_features = pocket_featurizer.featurize(
        protein_file, pockets, pocket_atoms, pocket_coords)
  
    assert isinstance(pocket_features, np.ndarray)
    assert pocket_features.shape[0] == n_pockets
