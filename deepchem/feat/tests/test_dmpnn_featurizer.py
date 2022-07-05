from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer, GraphConvConstants
from rdkit import Chem
import unittest
import numpy as np


class TestDMPNNFeaturizer(unittest.TestCase):
  """
  Test for DMPNN Featurizer class.
  """

  def setUp(self):
    """
    Set up tests.
    """
    self.smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C", "C"]
    self.features_generators = ['morgan']
    self.edge_index_orignal_order = [
        self.get_edge_index_original_order(smiles) for smiles in self.smiles
    ]
    self.edge_index_orignal_order_with_hs = [
        self.get_edge_index_original_order(smiles, is_add_hs=True)
        for smiles in self.smiles
    ]

  def get_edge_index_original_order(self, smiles, is_add_hs=False):
    """
    construct edge (bond) index.
    """
    mol = Chem.MolFromSmiles(smiles)
    if is_add_hs:
      mol = Chem.AddHs(mol)
    src, dest = [], []
    for bond in mol.GetBonds():
      # add edge list considering a directed graph
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      src += [start, end]
      dest += [end, start]
    return np.asarray([src, dest], dtype=int)

  def test_default_featurizer(self):
    """
    Test for featurization of 2 smiles using `DMPNNFeaturizer` class with no input parameters.
    """
    featurizer = DMPNNFeaturizer()
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (6,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].node_features_zero_padded.shape == (
        6 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].concatenated_features_zero_padded.shape == (
        12 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[0].mapping) == 12 + 1

    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order[0]).all()

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 22
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (22,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].node_features_zero_padded.shape == (
        22 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 44
    assert graph_feat[1].concatenated_features_zero_padded.shape == (
        44 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[1].mapping) == 44 + 1

    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order[1]).all()

    # assert "C"
    assert graph_feat[2].num_nodes == 1
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (1,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].node_features_zero_padded.shape == (
        1 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 0
    assert graph_feat[2].concatenated_features_zero_padded.shape == (
        1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[2].mapping) == 1

    assert (graph_feat[2].edge_index == self.edge_index_orignal_order[2]).all()

  def test_featurizer_with_original_atoms_ordering(self):
    """
    Test for featurization of 2 smiles using `DMPNNFeaturizer` class with `use_original_atom_ranks` = `True`.
    """
    featurizer = DMPNNFeaturizer(use_original_atom_ranks=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (6,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].node_features_zero_padded.shape == (
        6 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].concatenated_features_zero_padded.shape == (
        12 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[0].mapping) == 12 + 1

    assert (graph_feat[0].edge_index == self.edge_index_orignal_order[0]).all()

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 22
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (22,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].node_features_zero_padded.shape == (
        22 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 44
    assert graph_feat[1].concatenated_features_zero_padded.shape == (
        44 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[1].mapping) == 44 + 1

    assert (graph_feat[1].edge_index == self.edge_index_orignal_order[1]).all()

    # assert "C"
    assert graph_feat[2].num_nodes == 1
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (1,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].node_features_zero_padded.shape == (
        1 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 0
    assert graph_feat[2].concatenated_features_zero_padded.shape == (
        1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[2].mapping) == 1

    assert (graph_feat[2].edge_index == self.edge_index_orignal_order[2]).all()

  def test_featurizer_with_adding_hs(self):
    """
    Test for featurization of 2 smiles using `DMPNNFeaturizer` class with `is_adding_hs` set to `True`.
    """
    featurizer = DMPNNFeaturizer(is_adding_hs=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 11
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (11,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].node_features_zero_padded.shape == (
        11 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 22
    assert graph_feat[0].concatenated_features_zero_padded.shape == (
        22 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[0].mapping) == 22 + 1
    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order_with_hs[0]).all()

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 49
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (49,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].node_features_zero_padded.shape == (
        49 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 98
    assert graph_feat[1].concatenated_features_zero_padded.shape == (
        98 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[1].mapping) == 98 + 1
    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order_with_hs[1]).all()

    # assert "C"
    assert graph_feat[2].num_nodes == 5
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (5,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].node_features_zero_padded.shape == (
        5 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 8
    assert graph_feat[2].concatenated_features_zero_padded.shape == (
        8 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[2].mapping) == 8 + 1

    # for single C atom with hs, edge_index for original order is equal to edge_index for canonical order
    assert (graph_feat[2].edge_index == self.edge_index_orignal_order_with_hs[2]
           ).all()

  def test_featurizer_with_global_features(self):
    """
    Test for featurization of 2 smiles using `DMPNNFeaturizer` class with a given list of `features_generators`.
    """
    featurizer = DMPNNFeaturizer(features_generators=self.features_generators)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (6,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].node_features_zero_padded.shape == (
        6 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].concatenated_features_zero_padded.shape == (
        12 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[0].mapping) == 12 + 1
    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order[0]).all()

    assert len(graph_feat[0].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[0].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 9

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 22
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (22,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].node_features_zero_padded.shape == (
        22 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 44
    assert graph_feat[1].concatenated_features_zero_padded.shape == (
        44 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[1].mapping) == 44 + 1
    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order[1]).all()

    assert len(graph_feat[1].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[1].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 45

    # assert "C"
    assert graph_feat[2].num_nodes == 1
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (1,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].node_features_zero_padded.shape == (
        1 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 0
    assert graph_feat[2].concatenated_features_zero_padded.shape == (
        1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[2].mapping) == 1

    assert (graph_feat[2].edge_index == self.edge_index_orignal_order[2]).all()

    assert len(graph_feat[2].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[2].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 1

  def test_featurizer_with_adding_hs_and_global_features(self):
    """
    Test for featurization of 2 smiles using `DMPNNFeaturizer` class with `is_adding_hs` set to `True` and a given list of `features_generators`.
    """
    featurizer = DMPNNFeaturizer(is_adding_hs=True,
                                 features_generators=self.features_generators)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 11
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (11,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].node_features_zero_padded.shape == (
        11 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 22
    assert graph_feat[0].concatenated_features_zero_padded.shape == (
        22 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[0].mapping) == 22 + 1
    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order_with_hs[0]).all()

    assert len(graph_feat[0].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[0].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 10

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 49
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (49,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].node_features_zero_padded.shape == (
        49 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 98
    assert graph_feat[1].concatenated_features_zero_padded.shape == (
        98 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[1].mapping) == 98 + 1
    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order_with_hs[1]).all()

    assert len(graph_feat[1].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[1].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 57

    # assert "C"
    assert graph_feat[2].num_nodes == 5
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (5,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].node_features_zero_padded.shape == (
        5 + 1, GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 8
    assert graph_feat[2].concatenated_features_zero_padded.shape == (
        8 + 1, GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM)
    assert len(graph_feat[2].mapping) == 8 + 1

    # for single C atom with hs, edge_index for original order is equal to edge_index for canonical order
    assert (graph_feat[2].edge_index == self.edge_index_orignal_order_with_hs[2]
           ).all()

    assert len(graph_feat[2].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[2].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 4
