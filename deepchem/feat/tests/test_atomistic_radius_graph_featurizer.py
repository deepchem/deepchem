import numpy as np
import pytest

from ase import Atoms

import deepchem as dc


class TestAtomisticRadiusGraphFeaturizer:

    def test_two_atoms_inside_cutoff(self):
        featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=1.5)
        atoms = Atoms(numbers=[1, 8],
                      positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        graph = featurizer.featurize([atoms])[0]

        assert isinstance(graph, dc.feat.GraphData)
        np.testing.assert_array_equal(graph.node_features,
                                      np.array([[1], [8]], dtype=int))
        np.testing.assert_array_equal(graph.atomic_numbers,
                                      np.array([1, 8], dtype=int))
        np.testing.assert_allclose(graph.node_pos_features,
                                   np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        np.testing.assert_array_equal(graph.edge_index,
                                      np.array([[0, 1], [1, 0]], dtype=int))
        np.testing.assert_allclose(
            graph.edge_features, np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(graph.edge_distances, np.array([[1.0],
                                                                   [1.0]]))
        assert graph.edge_features.shape == (2, 3)
        assert graph.edge_distances.shape == (2, 1)

    def test_two_atoms_outside_cutoff(self):
        featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=0.5)
        atoms = Atoms(numbers=[1, 8],
                      positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        graph = featurizer.featurize([atoms])[0]

        assert graph.num_edges == 0
        assert graph.edge_index.shape == (2, 0)
        assert graph.edge_features.shape == (0, 3)
        assert graph.edge_distances.shape == (0, 1)

    def test_single_atom_no_edges(self):
        featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=1.0)
        atoms = Atoms(numbers=[6], positions=[[0.0, 0.0, 0.0]])

        graph = featurizer.featurize([atoms])[0]

        np.testing.assert_array_equal(graph.node_features,
                                      np.array([[6]], dtype=np.int64))
        assert graph.num_nodes == 1
        assert graph.num_edges == 0
        assert graph.edge_index.shape == (2, 0)
        assert graph.edge_features.shape == (0, 3)
        assert graph.edge_distances.shape == (0, 1)

    def test_three_atoms_deterministic_directed_edges(self):
        featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=1.1)
        atoms = Atoms(numbers=[1, 6, 8],
                      positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                 [0.0, 2.0, 0.0]])

        graph = featurizer.featurize([atoms])[0]

        np.testing.assert_array_equal(graph.edge_index,
                                      np.array([[0, 1], [1, 0]], dtype=int))
        np.testing.assert_allclose(
            graph.edge_features, np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

    def test_edge_displacements_and_distances(self):
        featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=2.0)
        atoms = Atoms(numbers=[1, 8],
                      positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        graph = featurizer.featurize([atoms])[0]
        expected_distance = np.sqrt(2.0)

        np.testing.assert_allclose(
            graph.edge_features, np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]))
        np.testing.assert_allclose(
            graph.edge_distances,
            np.array([[expected_distance], [expected_distance]]))
        assert graph.edge_distances.shape == (2, 1)

    def test_invalid_cutoff_raises(self):
        with pytest.raises(ValueError):
            dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=0.0)

    def test_invalid_input_type_raises(self):
        featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=1.0)

        with pytest.raises(TypeError):
            featurizer._featurize(np.array([[0.0, 0.0, 0.0]]))
