"""
Test Binding Pocket Features. 
"""
import os
import numpy as np
import unittest
import deepchem as dc


class TestBindingPocketFeatures(unittest.TestCase):
  """
  Test AtomicCoordinates.
  """

  def test_pocket_features(self):
    """
    Simple test that pocket_features return right shapes.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir,
                                "../../dock/tests/1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")

    finder = dc.dock.ConvexHullPocketFinder()
    pocket_featurizer = dc.feat.BindingPocketFeaturizer()
    pockets = finder.find_pockets(protein_file)
    n_pockets = len(pockets)

    pocket_features = pocket_featurizer.featurize(protein_file, pockets)

    assert isinstance(pocket_features, np.ndarray)
    assert pocket_features.shape[0] == n_pockets
