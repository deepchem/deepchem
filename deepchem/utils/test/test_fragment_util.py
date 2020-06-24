import os
import unittest
from deepchem.utils import rdkit_util
from deepchem.utils.fragment_util import get_contact_atom_indices
from deepchem.utils.fragment_util import MolecularFragment


class TestFragmentUtil(unittest.TestCase):

  def setUp(self):
    # TODO test more formats for ligand
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.protein_file = os.path.join(
        current_dir, '../../feat/tests/3ws9_protein_fixer_rdkit.pdb')
    self.ligand_file = os.path.join(current_dir,
                                    '../../feat/tests/3ws9_ligand.sdf')

  def test_get_contact_atom_indices(self):
    complexes = rdkit_util.load_complex([self.protein_file, self.ligand_file])
    contact_indices = get_contact_atom_indices(complexes)
    assert len(contact_indices) == 2

  def test_create_molecular_fragment(self):
    mol_xyz, mol_rdk = rdkit_util.load_molecule(self.ligand_file)
    fragment = MolecularFragment(mol_rdk.GetAtoms(), mol_xyz)
    assert len(mol_rdk.GetAtoms()) == len(fragment.GetAtoms())
    assert (fragment.GetCoords() == mol_xyz).all()
