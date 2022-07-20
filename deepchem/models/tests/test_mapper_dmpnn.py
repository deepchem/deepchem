from deepchem.models.torch_models.dmpnn import _MapperDMPNN
from deepchem.feat import DMPNNFeaturizer
import numpy as np
import unittest


class TestMapperDMPNN(unittest.TestCase):
  """
  Test for `_MapperDMPNN` helper class for DMPNN model
  """

  def setUp(self):
    """
    Set up tests.
    """
    smiles_list = ["C", "CC", "CCC", "C1=CC=CC=C1"]
    featurizer = DMPNNFeaturizer(use_original_atom_ranks=['morgan'])
    self.graphs = featurizer.featurize(smiles_list)
    self.benezene_mapping = np.asarray([[-1, 10], [-1, 3], [0, -1], [-1, 5],
                                        [2, -1], [-1, 7], [4, -1], [-1, 9],
                                        [6, -1], [-1, 11], [8, -1], [1, -1]])

  def test_general(self):
    """
    General tests for the mapper class
    """
    for graph in self.graphs:
      mapper = _MapperDMPNN(graph)
      assert (mapper.atom_features == graph.node_features).all()
      assert (mapper.bond_features == graph.edge_features).all()
      assert (mapper.bond_index == graph.edge_index).all()
      assert (mapper.global_features == graph.global_features).all()

      concat_feature_dim = graph.num_node_features + graph.num_edge_features
      assert mapper.f_ini_atoms_bonds.shape == (graph.num_edges + 1,
                                                concat_feature_dim)
      assert (
          mapper.f_ini_atoms_bonds[-1] == np.zeros(concat_feature_dim)).all()

      assert mapper.values == (mapper.f_ini_atoms_bonds, mapper.mapping,
                               mapper.global_features)

  def test_mapper_no_bond(self):
    """
    Test 'C' in _MapperDMPNN (no bond present)
    """
    mapper = _MapperDMPNN(self.graphs[0])
    assert (mapper.bond_to_ini_atom == np.empty(0)).all()
    assert (mapper.mapping == np.asarray([[-1]])).all()

  def test_mapper_two_directed_bonds_btw_two_atoms(self):
    """
    Test 'CC' in _MapperDMPNN (1 bond present (2 directed))
    """
    mapper = _MapperDMPNN(self.graphs[1])
    assert (mapper.bond_to_ini_atom == np.asarray([0, 1])).all()
    assert (mapper.mapping == np.asarray([[-1], [-1]])).all()

  def test_mapper_two_adjacent_bonds(self):
    """
    Test 'CCC' in _MapperDMPNN (2 adjacent bonds present (4 directed))
    """
    mapper = _MapperDMPNN(self.graphs[2])
    assert (mapper.bond_to_ini_atom == np.asarray([0, 1, 1, 2])).all()
    assert (mapper.mapping == np.asarray([[-1, -1], [-1, 3], [0, -1],
                                          [-1, -1]])).all()

  def test_mapper_ring(self):
    """
    Test 'C1=CC=CC=C1' in _MapperDMPNN (benezene ring)
    """
    mapper = _MapperDMPNN(self.graphs[3])
    assert (mapper.bond_to_ini_atom == np.asarray(
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0])).all()
    assert (mapper.mapping == self.benezene_mapping).all()
