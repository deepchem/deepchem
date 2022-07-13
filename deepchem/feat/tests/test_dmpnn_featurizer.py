from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer, GraphConvConstants
import unittest
import numpy as np

edge_index_orignal_order = {
    "C1=CC=NC=C1":
        np.asarray([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]),
    "CC(=O)C":
        np.asarray([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]]),
    "C":
        np.empty((2, 0), dtype=int)
}

edge_index_orignal_order_with_hs = {
    "C1=CC=NC=C1":
        np.asarray([[
            0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 6, 1, 7, 2, 8, 4, 9, 5, 10
        ], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5, 6, 0, 7, 1, 8, 2, 9, 4, 10,
            5]]),
    "CC(=O)C":
        np.asarray([[0, 1, 1, 2, 1, 3, 0, 4, 0, 5, 0, 6, 3, 7, 3, 8, 3, 9],
                    [1, 0, 2, 1, 3, 1, 4, 0, 5, 0, 6, 0, 7, 3, 8, 3, 9, 3]]),
    "C":
        np.asarray([[0, 1, 0, 2, 0, 3, 0, 4], [1, 0, 2, 0, 3, 0, 4, 0]])
}


class TestDMPNNFeaturizer(unittest.TestCase):
  """
  Test for DMPNN Featurizer class.
  """

  def setUp(self):
    """
    Set up tests.
    """
    self.smiles = ["C1=CC=NC=C1", "CC(=O)C", "C"]
    self.features_generators = ['morgan']
    self.edge_index_orignal_order = list(edge_index_orignal_order.values())
    self.edge_index_orignal_order_with_hs = list(
        edge_index_orignal_order_with_hs.values())

  def test_default_featurizer_ring(self):
    """
    Test for featurization of "C1=CC=NC=C1" using `DMPNNFeaturizer` class with no input parameters.
    """
    featurizer = DMPNNFeaturizer()
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=NC=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (6,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (12,
                                                 GraphConvConstants.BOND_FDIM)

    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order[0]).all()

  def test_default_featurizer_general_case(self):
    """
    Test for featurization of "CC(=O)C" using `DMPNNFeaturizer` class with no input parameters.
    """
    featurizer = DMPNNFeaturizer()
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "CC(=O)C"
    assert graph_feat[1].num_nodes == 4
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (4,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 6
    assert graph_feat[1].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[1].edge_features.shape == (6,
                                                 GraphConvConstants.BOND_FDIM)

    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order[1]).all()

  def test_default_featurizer_single_atom(self):
    """
    Test for featurization of "C" using `DMPNNFeaturizer` class with no input parameters.
    """
    featurizer = DMPNNFeaturizer()
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C"
    assert graph_feat[2].num_nodes == 1
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (1,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 0
    assert graph_feat[2].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[2].edge_features.shape == (0,
                                                 GraphConvConstants.BOND_FDIM)

    assert (graph_feat[2].edge_index == self.edge_index_orignal_order[2]).all()

  def test_featurizer_with_original_atoms_ordering_ring(self):
    """
    Test for featurization of "C1=CC=CN=C1" using `DMPNNFeaturizer` class with `use_original_atom_ranks` = `True`.
    """
    featurizer = DMPNNFeaturizer(use_original_atom_ranks=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (6,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (12,
                                                 GraphConvConstants.BOND_FDIM)

    assert (graph_feat[0].edge_index == self.edge_index_orignal_order[0]).all()

  def test_featurizer_with_original_atoms_ordering_general_case(self):
    """
    Test for featurization of "CC(=O)C" using `DMPNNFeaturizer` class with `use_original_atom_ranks` = `True`.
    """
    featurizer = DMPNNFeaturizer(use_original_atom_ranks=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "CC(=O)C"
    assert graph_feat[1].num_nodes == 4
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (4,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 6
    assert graph_feat[1].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[1].edge_features.shape == (6,
                                                 GraphConvConstants.BOND_FDIM)

    assert (graph_feat[1].edge_index == self.edge_index_orignal_order[1]).all()

  def test_featurizer_with_original_atoms_ordering_single_atom(self):
    """
    Test for featurization of "C" using `DMPNNFeaturizer` class with `use_original_atom_ranks` = `True`.
    """
    featurizer = DMPNNFeaturizer(use_original_atom_ranks=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C"
    assert graph_feat[2].num_nodes == 1
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (1,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 0
    assert graph_feat[2].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[2].edge_features.shape == (0,
                                                 GraphConvConstants.BOND_FDIM)

    assert (graph_feat[2].edge_index == self.edge_index_orignal_order[2]).all()

  def test_featurizer_with_adding_hs_ring(self):
    """
    Test for featurization of "C1=CC=CN=C1" using `DMPNNFeaturizer` class with `is_adding_hs` set to `True`.
    """
    featurizer = DMPNNFeaturizer(is_adding_hs=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 11
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (11,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 22
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (22,
                                                 GraphConvConstants.BOND_FDIM)
    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order_with_hs[0]).all()

  def test_featurizer_with_adding_hs_general_case(self):
    """
    Test for featurization of "CC(=O)C" using `DMPNNFeaturizer` class with `is_adding_hs` set to `True`.
    """
    featurizer = DMPNNFeaturizer(is_adding_hs=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "CC(=O)C"
    assert graph_feat[1].num_nodes == 10
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (10,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 18
    assert graph_feat[1].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[1].edge_features.shape == (18,
                                                 GraphConvConstants.BOND_FDIM)
    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order_with_hs[1]).all()

  def test_featurizer_with_adding_hs_single_atom(self):
    """
    Test for featurization of "C" using `DMPNNFeaturizer` class with `is_adding_hs` set to `True`.
    """
    featurizer = DMPNNFeaturizer(is_adding_hs=True)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C"
    assert graph_feat[2].num_nodes == 5
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (5,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 8
    assert graph_feat[2].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[2].edge_features.shape == (8,
                                                 GraphConvConstants.BOND_FDIM)

    # for single C atom with hs, edge_index for original order is equal to edge_index for canonical order
    assert (graph_feat[2].edge_index == self.edge_index_orignal_order_with_hs[2]
           ).all()

  def test_featurizer_with_global_features_ring(self):
    """
    Test for featurization of "C1=CC=CN=C1" using `DMPNNFeaturizer` class with a given list of `features_generators`.
    """
    featurizer = DMPNNFeaturizer(features_generators=self.features_generators)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[0].node_features.shape == (6,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (12,
                                                 GraphConvConstants.BOND_FDIM)

    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order[0]).all()

    assert len(graph_feat[0].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[0].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 9

  def test_featurizer_with_global_features_general_case(self):
    """
    Test for featurization of "CC(=O)C" using `DMPNNFeaturizer` class with a given list of `features_generators`.
    """
    featurizer = DMPNNFeaturizer(features_generators=self.features_generators)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "CC(=O)C"
    assert graph_feat[1].num_nodes == 4
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (4,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 6
    assert graph_feat[1].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[1].edge_features.shape == (6,
                                                 GraphConvConstants.BOND_FDIM)

    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order[1]).all()

    assert len(graph_feat[1].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[1].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 6

  def test_featurizer_with_global_features_single_atom(self):
    """
    Test for featurization of "C" using `DMPNNFeaturizer` class with a given list of `features_generators`.
    """
    featurizer = DMPNNFeaturizer(features_generators=self.features_generators)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C"
    assert graph_feat[2].num_nodes == 1
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (1,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 0
    assert graph_feat[2].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[2].edge_features.shape == (0,
                                                 GraphConvConstants.BOND_FDIM)

    assert (graph_feat[2].edge_index == self.edge_index_orignal_order[2]).all()

    assert len(graph_feat[2].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[2].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 1

  def test_featurizer_with_adding_hs_and_global_features_ring(self):
    """
    Test for featurization of "C1=CC=CN=C1" using `DMPNNFeaturizer` class with `is_adding_hs` set to `True` and a given list of `features_generators`.
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
    assert graph_feat[0].num_edges == 22
    assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[0].edge_features.shape == (22,
                                                 GraphConvConstants.BOND_FDIM)
    assert not (graph_feat[0].edge_index
                == self.edge_index_orignal_order_with_hs[0]).all()

    assert len(graph_feat[0].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[0].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 10

  def test_featurizer_with_adding_hs_and_global_features_general_case(self):
    """
    Test for featurization of "CC(=O)C" using `DMPNNFeaturizer` class with `is_adding_hs` set to `True` and a given list of `features_generators`.
    """
    featurizer = DMPNNFeaturizer(is_adding_hs=True,
                                 features_generators=self.features_generators)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "CC(=O)C"
    assert graph_feat[1].num_nodes == 10
    assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[1].node_features.shape == (10,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[1].num_edges == 18
    assert graph_feat[1].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[1].edge_features.shape == (18,
                                                 GraphConvConstants.BOND_FDIM)
    assert not (graph_feat[1].edge_index
                == self.edge_index_orignal_order_with_hs[1]).all()

    assert len(graph_feat[1].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[1].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 10

  def test_featurizer_with_adding_hs_and_global_features_single_atom(self):
    """
    Test for featurization of "C" using `DMPNNFeaturizer` class with `is_adding_hs` set to `True` and a given list of `features_generators`.
    """
    featurizer = DMPNNFeaturizer(is_adding_hs=True,
                                 features_generators=self.features_generators)
    graph_feat = featurizer.featurize(self.smiles)
    assert len(graph_feat) == 3

    # assert "C"
    assert graph_feat[2].num_nodes == 5
    assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
    assert graph_feat[2].node_features.shape == (5,
                                                 GraphConvConstants.ATOM_FDIM)
    assert graph_feat[2].num_edges == 8
    assert graph_feat[2].num_edge_features == GraphConvConstants.BOND_FDIM
    assert graph_feat[2].edge_features.shape == (8,
                                                 GraphConvConstants.BOND_FDIM)

    # for single C atom with hs, edge_index for original order is equal to edge_index for canonical order
    assert (graph_feat[2].edge_index == self.edge_index_orignal_order_with_hs[2]
           ).all()

    assert len(graph_feat[2].global_features
              ) == 2048  # for `morgan` features generator
    nonzero_features_indicies = graph_feat[2].global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 4
