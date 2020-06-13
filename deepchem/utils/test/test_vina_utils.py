"""
Test Autodock Vina Utility Functions.
"""
import os
import numpy as np
import unittest
from deepchem.utils import vina_utils
from deepchem.utils import rdkit_util


class TestVinaUtils(unittest.TestCase):

  def setUp(self):
    # TODO test more formats for ligand
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.docked_ligands = os.path.join(current_dir, '1jld_ligand_docked.pdbqt')

  def test_load_docked_ligand(self):
    docked_ligands, scores = vina_utils.load_docked_ligands(self.docked_ligands)
    assert len(docked_ligands) == 9
    assert len(scores) == 9

    for ligand, score in zip(docked_ligands, scores):
      xyz = rdkit_util.get_xyz_from_mol(ligand)
      assert score < 0  # This is a binding free energy
      assert np.count_nonzero(xyz) > 0
