from deepchem.feat.molecule_featurizers.dmpnn_featurizer import _MapperDMPNN, GraphConvConstants, atom_features
from rdkit import Chem
import numpy as np
import unittest


class TestMapperDMPNN(unittest.TestCase):
  """
  Test for `MapperDMPNN` helper class for DMPNN featurizer
  """

  def setUp(self):
    """
    Set up tests.
    """
    atom_fdim = GraphConvConstants.ATOM_FDIM
    bond_fdim = GraphConvConstants.BOND_FDIM
    self.concat_fdim = atom_fdim + bond_fdim

    smiles_list = ["C", "CC", "CCC", "C1=CC=CC=C1"]
    self.mol = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    self.features = [
        self.get_f_atoms_zero_padded(mol, atom_fdim) for mol in self.mol
    ]
    self.benezene_mapping = np.asarray([[0, 0], [0, 4], [0, 6], [2, 0], [0, 11],
                                        [1, 0], [0, 8], [5, 0], [0, 10], [7, 0],
                                        [0, 12], [9, 0], [3, 0]])

  def get_f_atoms_zero_padded(self, mol, atom_fdim):
    """
    Helper method to find atoms features list (zero-padded) from the given RDKit mol
    """
    # get atom features
    f_atoms = np.asarray([atom_features(atom) for atom in mol.GetAtoms()],
                         dtype=float)

    # mapping from atom index to atom features | initial input is a zero padding
    f_atoms_zero_padded = np.asarray([[0] * atom_fdim], dtype=float)
    f_atoms_zero_padded = np.concatenate((f_atoms_zero_padded, f_atoms), axis=0)
    return f_atoms_zero_padded

  def test_mapper_no_bond(self):
    """
    Test 'C' in _MapperDMPNN (no bond present)
    """
    mapper = _MapperDMPNN(self.mol[0], self.concat_fdim, self.features[0])
    assert mapper.num_atoms == 1
    assert mapper.num_bonds == 0
    assert len(mapper.f_atoms_zero_padded) == 2
    assert len(mapper.f_ini_atoms_bonds_zero_padded) == 1
    assert mapper.atom_to_incoming_bonds == [[0], [0]]
    assert mapper.bond_to_ini_atom == [0]
    assert mapper.b2revb == [0]
    assert (mapper.mapping == np.asarray([[0]])).all()

  def test_mapper_two_directed_bonds_btw_two_atoms(self):
    """
    Test 'CC' in _MapperDMPNN (1 bond present (2 directed))
    """
    mapper = _MapperDMPNN(self.mol[1], self.concat_fdim, self.features[1])
    assert mapper.num_atoms == 2
    assert mapper.num_bonds == 2
    assert len(mapper.f_atoms_zero_padded) == 3
    assert len(mapper.f_ini_atoms_bonds_zero_padded) == 3
    assert mapper.atom_to_incoming_bonds == [[0], [2], [1]]
    assert mapper.bond_to_ini_atom == [0, 1, 2]
    assert mapper.b2revb == [0, 2, 1]
    assert (mapper.mapping == np.asarray([[0], [0], [0]])).all()

  def test_mapper_two_adjacent_bonds(self):
    """
    Test 'CCC' in _MapperDMPNN (2 adjacent bonds present (4 directed))
    """
    mapper = _MapperDMPNN(self.mol[2], self.concat_fdim, self.features[2])
    assert mapper.num_atoms == 3
    assert mapper.num_bonds == 4
    assert len(mapper.f_atoms_zero_padded) == 4
    assert len(mapper.f_ini_atoms_bonds_zero_padded) == 5
    assert mapper.atom_to_incoming_bonds == [[0, 0], [2, 0], [1, 4], [3, 0]]
    assert mapper.bond_to_ini_atom == [0, 1, 2, 2, 3]
    assert mapper.b2revb == [0, 2, 1, 4, 3]
    assert (mapper.mapping == np.asarray([[0, 0], [0, 0], [0, 4], [1, 0],
                                          [0, 0]])).all()

  def test_mapper_ring(self):
    """
    Test 'C1=CC=CN=C1' in _MapperDMPNN (benezene ring)
    """
    mapper = _MapperDMPNN(self.mol[3], self.concat_fdim, self.features[3])
    assert mapper.num_atoms == 6
    assert mapper.num_bonds == 12
    assert len(mapper.f_atoms_zero_padded) == 7
    assert len(mapper.f_ini_atoms_bonds_zero_padded) == 13
    assert mapper.atom_to_incoming_bonds == [[0, 0], [2, 4], [1, 6], [5, 8],
                                             [7, 10], [9, 12], [3, 11]]
    assert mapper.bond_to_ini_atom == [0, 1, 2, 1, 6, 2, 3, 3, 4, 4, 5, 5, 6]
    assert mapper.b2revb == [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
    assert (mapper.mapping == self.benezene_mapping).all()
