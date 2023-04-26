"""
Test featurizers for inorganic crystals.
"""
import unittest
import numpy as np

from deepchem.feat import ElementPropertyFingerprint, SineCoulombMatrix, CGCNNFeaturizer, ElemNetFeaturizer


class TestMaterialFeaturizers(unittest.TestCase):
    """
    Test material featurizers.
    """

    def setUp(self):
        """
        Set up tests.
        """
        self.formula = 'MoS2'
        self.struct_dict = {
            '@module':
                'pymatgen.core.structure',
            '@class':
                'Structure',
            'charge':
                None,
            'lattice': {
                'matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                'a': 1.0,
                'b': 1.0,
                'c': 1.0,
                'alpha': 90.0,
                'beta': 90.0,
                'gamma': 90.0,
                'volume': 1.0
            },
            'sites': [{
                'species': [{
                    'element': 'Fe',
                    'occu': 1
                }],
                'abc': [0.0, 0.0, 0.0],
                'xyz': [0.0, 0.0, 0.0],
                'label': 'Fe',
                'properties': {}
            }]
        }

    def test_element_property_fingerprint(self):
        """
        Test Element Property featurizer.
        """

        featurizer = ElementPropertyFingerprint(data_source='matminer')
        features = featurizer.featurize([self.formula])

        assert len(features[0]) == 65
        assert np.allclose(features[0][:5],
                           [2.16, 2.58, 0.42, 2.44, 0.29698485],
                           atol=0.1)

    def test_sine_coulomb_matrix(self):
        """
        Test SCM featurizer.
        """

        featurizer = SineCoulombMatrix(max_atoms=3)
        features = featurizer.featurize([self.struct_dict])

        assert len(features) == 1
        assert features.shape == (1, 3)
        assert np.isclose(features[0][0], 1244, atol=.5)

    def test_cgcnn_featurizer(self):
        """
        Test CGCNNFeaturizer.
        """

        featurizer = CGCNNFeaturizer(radius=3.0, max_neighbors=6, step=0.3)
        graph_features = featurizer.featurize([self.struct_dict])

        assert graph_features[0].num_nodes == 1
        assert graph_features[0].num_edges == 6
        assert graph_features[0].node_features.shape == (1, 92)
        assert graph_features[0].edge_index.shape == (2, 6)
        assert graph_features[0].edge_features.shape == (6, 11)

    def test_elemnet_featurizer(self):
        """
        Test ElemNetFeaturizer.
        """

        featurizer = ElemNetFeaturizer()
        features = featurizer.featurize([self.formula])

        assert features.shape[1] == 86
        assert np.isclose(features[0][13], 0.6666667, atol=0.01)
        assert np.isclose(features[0][38], 0.33333334, atol=0.01)
        assert np.isclose(features.sum(), 1.0, atol=0.01)
