import numpy as np
import unittest
from deepchem.utils import voxel_utils
from deepchem.utils import hash_utils


class TestVoxelUtils(unittest.TestCase):

    def test_voxelize_vector_features(self):
        coordinates = np.array([[0., 0., 0.], [0.1, 0.1, 0.1], [10., 10., 10.]])
        for feature_type in (list, np.array):
            with self.subTest(feature_type=feature_type):
                features = voxel_utils.voxelize(
                    voxel_utils.convert_atom_to_voxel,
                    coordinates,
                    box_width=2,
                    feature_dict={
                        0: feature_type([1., 2., 3., 4.]),
                        1: feature_type([4., 5., 6., 7.]),
                        2: feature_type([7., 8., 9., 10.]),
                    },
                    nb_channel=4,
                    dtype='float')
                expected = np.zeros((2, 2, 2, 4))
                expected[1, 1, 1] = [5., 7., 9., 11.]
                np.testing.assert_array_equal(features, expected)

    def test_voxelize_atom_pair_vector_features(self):
        coordinates = (np.array([[0., 0., 0.]]), np.array([[-1., -1., -1.]]))
        features = voxel_utils.voxelize(voxel_utils.convert_atom_pair_to_voxel,
                                        coordinates,
                                        box_width=2,
                                        feature_dict={(0, 0): [2, 3]},
                                        nb_channel=2)
        expected = np.zeros((2, 2, 2, 2))
        expected[0, 0, 0] = [2, 3]
        expected[1, 1, 1] = [2, 3]
        np.testing.assert_array_equal(features, expected)

    def test_voxelize_scalar_features(self):
        for nb_channel in (1, 4):
            with self.subTest(nb_channel=nb_channel):
                features = voxel_utils.voxelize(
                    voxel_utils.convert_atom_to_voxel,
                    np.zeros((1, 3)),
                    box_width=2,
                    feature_dict={0: 2.5},
                    nb_channel=nb_channel,
                    dtype='float')
                expected = np.zeros((2, 2, 2, nb_channel))
                expected[1, 1, 1, 0] = 2.5
                np.testing.assert_array_equal(features, expected)

    def test_convert_atom_to_voxel(self):
        N = 5
        coordinates = np.random.rand(N, 3)
        atom_index = 2
        box_width = 16
        voxel_width = 1
        indices = voxel_utils.convert_atom_to_voxel(coordinates, atom_index,
                                                    box_width, voxel_width)
        assert indices.shape == (3,)

    def test_convert_pair_atom_to_voxel(self):
        N = 5
        M = 6
        coordinates1 = np.random.rand(N, 3)
        coordinates2 = np.random.rand(M, 3)
        atom_index_pair = (2, 3)
        box_width = 16
        voxel_width = 1
        indices = voxel_utils.convert_atom_pair_to_voxel(
            [coordinates1, coordinates2], atom_index_pair, box_width,
            voxel_width)
        assert indices.shape == (2, 3)

    def test_voxelize_convert_atom(self):
        N = 5
        coordinates = np.random.rand(N, 3)
        box_width = 16
        voxel_width = 1
        voxels_per_edge = int(box_width / voxel_width)
        get_voxels = voxel_utils.convert_atom_to_voxel
        hash_function = hash_utils.hash_ecfp
        feature_dict = {1: "C", 2: "CC"}
        nb_channel = 16
        features = voxel_utils.voxelize(get_voxels,
                                        coordinates,
                                        box_width,
                                        voxel_width,
                                        hash_function,
                                        feature_dict,
                                        nb_channel=nb_channel)
        assert features.shape == (voxels_per_edge, voxels_per_edge,
                                  voxels_per_edge, nb_channel)

    def test_voxelize_convert_atom_pair(self):
        N = 5
        M = 6
        coordinates1 = np.random.rand(N, 3)
        coordinates2 = np.random.rand(M, 3)
        coordinates = [coordinates1, coordinates2]
        box_width = 16
        voxel_width = 1
        voxels_per_edge = int(box_width / voxel_width)
        get_voxels = voxel_utils.convert_atom_pair_to_voxel
        hash_function = hash_utils.hash_ecfp_pair
        feature_dict = {(1, 2): ("C", "O"), (2, 3): ("CC", "OH")}
        nb_channel = 16
        features = voxel_utils.voxelize(get_voxels,
                                        coordinates,
                                        box_width,
                                        voxel_width,
                                        hash_function,
                                        feature_dict,
                                        nb_channel=nb_channel)
        assert features.shape == (voxels_per_edge, voxels_per_edge,
                                  voxels_per_edge, nb_channel)
