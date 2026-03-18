import os
import unittest
import numpy as np
from deepchem.utils import rdkit_utils
from deepchem.utils.fragment_utils import get_contact_atom_indices
from deepchem.utils.fragment_utils import merge_molecular_fragments
from deepchem.utils.fragment_utils import get_partial_charge
from deepchem.utils.fragment_utils import strip_hydrogens
from deepchem.utils.fragment_utils import MolecularFragment
from deepchem.utils.fragment_utils import AtomShim


class TestFragmentUtil(unittest.TestCase):

    def setUp(self):
        # TODO test more formats for ligand
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.protein_file = os.path.join(
            current_dir, '../../feat/tests/data/3ws9_protein_fixer_rdkit.pdb')
        self.ligand_file = os.path.join(
            current_dir, '../../feat/tests/data/3ws9_ligand.sdf')

    def test_get_contact_atom_indices(self):
        complexes = rdkit_utils.load_complex(
            [self.protein_file, self.ligand_file])
        contact_indices = get_contact_atom_indices(complexes)
        assert len(contact_indices) == 2

    def test_create_molecular_fragment(self):
        mol_xyz, mol_rdk = rdkit_utils.load_molecule(self.ligand_file)
        fragment = MolecularFragment(mol_rdk.GetAtoms(), mol_xyz)
        assert len(mol_rdk.GetAtoms()) == len(fragment.GetAtoms())
        assert (fragment.GetCoords() == mol_xyz).all()

    def test_strip_hydrogens(self):
        mol_xyz, mol_rdk = rdkit_utils.load_molecule(self.ligand_file)
        _ = MolecularFragment(mol_rdk.GetAtoms(), mol_xyz)

        # Test on RDKit
        _ = strip_hydrogens(mol_xyz, mol_rdk)

    def test_merge_molecular_fragments(self):
        mol_xyz, mol_rdk = rdkit_utils.load_molecule(self.ligand_file)
        fragment1 = MolecularFragment(mol_rdk.GetAtoms(), mol_xyz)
        fragment2 = MolecularFragment(mol_rdk.GetAtoms(), mol_xyz)
        joint = merge_molecular_fragments([fragment1, fragment2])
        assert len(mol_rdk.GetAtoms()) * 2 == len(joint.GetAtoms())

    def test_get_partial_charge(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC")
        atom = mol.GetAtoms()[0]
        partial_charge = get_partial_charge(atom)
        assert partial_charge == 0

    def test_atom_shim(self):
        atomic_num = 5
        partial_charge = 1
        atom_coords = np.array([0., 1., 2.])
        shim = AtomShim(atomic_num, partial_charge, atom_coords)
        assert shim.GetAtomicNum() == atomic_num
        assert shim.GetPartialCharge() == partial_charge
        assert (shim.GetCoords() == atom_coords).all()
