import unittest
import deepchem as dc

class TestSplifFingerprints(unittest.TestCase):
  """Test Splif Fingerprint and Voxelizer."""

  def setUp(self):
    # TODO test more formats for ligand
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.protein_file = os.path.join(current_dir,
                                     '3ws9_protein_fixer_rdkit.pdb')
    self.ligand_file = os.path.join(current_dir, '3ws9_ligand.sdf')
    self.complex_files = [(self.protein_file, self.ligand_file)]

