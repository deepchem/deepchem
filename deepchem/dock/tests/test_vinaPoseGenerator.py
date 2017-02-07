import unittest

from dock import VinaPoseGenerator


class TestVinaPoseGenerator(unittest.TestCase):
  def test_generate_poses(self):
    pose_generator = VinaPoseGenerator(exhaustiveness=1, detect_pockets=False)
    ligand_file = 'deepchem/dock/tests/1jld_ligand.sdf'
    protein_file = 'deepchem/dock/tests/1jld_protein.pdb'
    pose_generator.generate_poses(protein_file, ligand_file)
