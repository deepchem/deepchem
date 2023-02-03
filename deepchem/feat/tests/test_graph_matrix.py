import unittest
import numpy as np
from deepchem.feat.molecule_featurizers import GraphMatrix


class TestGraphMatrix(unittest.TestCase):

    def test_graph_matrix(self):

        max_atom_count = 5
        atom_array = [7, 7, 7, 8, 8, 8, 9, 6]

        A = np.zeros(shape=(max_atom_count, max_atom_count), dtype=np.float32)
        X = np.array(atom_array, dtype=np.int32)

        graph_matrix = GraphMatrix(adjacency_matrix=A, node_features=X)
        assert isinstance(graph_matrix.adjacency_matrix, np.ndarray)
        assert isinstance(graph_matrix.node_features, np.ndarray)
        assert graph_matrix.adjacency_matrix.dtype == np.float32
        assert graph_matrix.node_features.dtype == np.int32
        assert graph_matrix.adjacency_matrix.shape == A.shape
        assert graph_matrix.node_features.shape == X.shape


if __name__ == '__main__':
    unittest.main()
