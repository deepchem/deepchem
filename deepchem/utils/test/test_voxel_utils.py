import numpy as np
import unittest
from deepchem.utils import voxel_utils
from deepchem.utils import hash_utils


class TestVoxelUtils(unittest.TestCase):

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
