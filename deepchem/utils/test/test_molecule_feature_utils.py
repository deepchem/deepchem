import unittest

from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_graph_distance_one_hot


class TestGraphConvUtils(unittest.TestCase):

  def setUp(self):
    from rdkit import Chem
    self.mol = Chem.MolFromSmiles("CN=C=O")  # methyl isocyanate
    self.mol_copper_sulfate = Chem.MolFromSmiles("[Cu+2].[O-]S(=O)(=O)[O-]")
    self.mol_benzene = Chem.MolFromSmiles("c1ccccc1")
    self.mol_s_alanine = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")

  def test_one_hot_encode(self):
    # string set
    assert one_hot_encode("a", ["a", "b", "c"]) == [1.0, 0.0, 0.0]
    # integer set
    assert one_hot_encode(2, [0.0, 1, 2]) == [0.0, 0.0, 1.0]
    # include_unknown_set is False
    assert one_hot_encode(3, [0.0, 1, 2]) == [0.0, 0.0, 0.0]
    # include_unknown_set is True
    assert one_hot_encode(3, [0.0, 1, 2], True) == [0.0, 0.0, 0.0, 1.0]

  def test_get_atom_type_one_hot(self):
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    one_hot = get_atom_type_one_hot(atoms[0])
    assert one_hot == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # check unknown atoms
    atoms = self.mol_copper_sulfate.GetAtoms()
    assert atoms[0].GetSymbol() == "Cu"
    one_hot = get_atom_type_one_hot(atoms[0])
    assert one_hot == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    one_hot = get_atom_type_one_hot(atoms[0], include_unknown_set=False)
    assert one_hot == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # check original set
    atoms = self.mol.GetAtoms()
    assert atoms[1].GetSymbol() == "N"
    original_set = ["C", "O", "N"]
    one_hot = get_atom_type_one_hot(atoms[1], allowable_set=original_set)
    assert one_hot == [0.0, 0.0, 1.0, 0.0]

  def test_construct_hydrogen_bonding_info(self):
    info = construct_hydrogen_bonding_info(self.mol)
    assert isinstance(info, list)
    assert isinstance(info[0], tuple)
    # Generally, =O behaves as an electron acceptor
    assert info[0] == (3, "Acceptor")

  def test_get_atom_hydrogen_bonding_one_hot(self):
    info = construct_hydrogen_bonding_info(self.mol)
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    one_hot = get_atom_hydrogen_bonding_one_hot(atoms[0], info)
    assert one_hot == [0.0, 0.0]

    assert atoms[3].GetSymbol() == "O"
    one_hot = get_atom_hydrogen_bonding_one_hot(atoms[3], info)
    assert one_hot == [0.0, 1.0]

  def test_get_atom_is_in_aromatic_one_hot(self):
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    one_hot = get_atom_is_in_aromatic_one_hot(atoms[0])
    assert one_hot == [0.0]

    atoms = self.mol_benzene.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    one_hot = get_atom_is_in_aromatic_one_hot(atoms[0])
    assert one_hot == [1.0]

  def test_get_atom_hybridization_one_hot(self):
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    one_hot = get_atom_hybridization_one_hot(atoms[0])
    assert one_hot == [0.0, 0.0, 1.0]

  def test_get_atom_total_num_Hs_one_hot(self):
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    one_hot = get_atom_total_num_Hs_one_hot(atoms[0])
    assert one_hot == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert atoms[3].GetSymbol() == "O"
    one_hot = get_atom_total_num_Hs_one_hot(atoms[3])
    assert one_hot == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  def test_get_atom_chirality_one_hot(self):
    atoms = self.mol_s_alanine.GetAtoms()
    assert atoms[0].GetSymbol() == "N"
    one_hot = get_atom_chirality_one_hot(atoms[0])
    assert one_hot == [0.0, 0.0]
    assert atoms[1].GetSymbol() == "C"
    one_hot = get_atom_chirality_one_hot(atoms[1])
    assert one_hot == [0.0, 1.0]

  def test_get_atom_formal_charge(self):
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    formal_charge = get_atom_formal_charge(atoms[0])
    assert formal_charge == [0.0]

  def test_get_atom_partial_charge(self):
    from rdkit.Chem import AllChem
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    with self.assertRaises(KeyError):
      get_atom_partial_charge(atoms[0])

    # we must compute partial charges before using `get_atom_partial_charge`
    AllChem.ComputeGasteigerCharges(self.mol)
    partial_charge = get_atom_partial_charge(atoms[0])
    assert len(partial_charge) == 1.0
    assert isinstance(partial_charge[0], float)

  def test_get_atom_total_degree_one_hot(self):
    atoms = self.mol.GetAtoms()
    assert atoms[0].GetSymbol() == "C"
    one_hot = get_atom_total_degree_one_hot(atoms[0])
    assert one_hot == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    assert atoms[3].GetSymbol() == "O"
    one_hot = get_atom_total_degree_one_hot(atoms[3])
    assert one_hot == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  def test_get_bond_type_one_hot(self):
    bonds = self.mol.GetBonds()
    one_hot = get_bond_type_one_hot(bonds[0])
    # The C-N bond is a single bond
    assert bonds[0].GetBeginAtomIdx() == 0.0
    assert bonds[0].GetEndAtomIdx() == 1.0
    assert one_hot == [1.0, 0.0, 0.0, 0.0]

  def test_get_bond_is_in_same_ring_one_hot(self):
    bonds = self.mol.GetBonds()
    one_hot = get_bond_is_in_same_ring_one_hot(bonds[0])
    assert one_hot == [0.0]

    bonds = self.mol_benzene.GetBonds()
    one_hot = get_bond_is_in_same_ring_one_hot(bonds[0])
    assert one_hot == [1.0]

  def test_get_bond_is_conjugated_one_hot(self):
    bonds = self.mol.GetBonds()
    one_hot = get_bond_is_conjugated_one_hot(bonds[0])
    assert one_hot == [0.0]

    bonds = self.mol_benzene.GetBonds()
    one_hot = get_bond_is_conjugated_one_hot(bonds[0])
    assert one_hot == [1.0]

  def test_get_bond_stereo_one_hot(self):
    bonds = self.mol.GetBonds()
    one_hot = get_bond_stereo_one_hot(bonds[0])
    assert one_hot == [1.0, 0.0, 0.0, 0.0, 0.0]

  def test_get_bond_graph_distance_one_hot(self):
    from rdkit import Chem
    bonds = self.mol.GetBonds()
    dist_matrix = Chem.GetDistanceMatrix(self.mol)
    one_hot = get_bond_graph_distance_one_hot(bonds[0], dist_matrix)
    assert one_hot == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
