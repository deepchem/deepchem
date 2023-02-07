import unittest
import os
import deepchem as dc


class TestSplifFingerprints(unittest.TestCase):
    """Test Splif Fingerprint and Voxelizer."""

    def setUp(self):
        # TODO test more formats for ligand
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.protein_file = os.path.join(current_dir, 'data',
                                         '3ws9_protein_fixer_rdkit.pdb')
        self.ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')
        self.complex_files = [(self.ligand_file, self.protein_file)]

    def test_splif_shape(self):
        size = 8
        featurizer = dc.feat.SplifFingerprint(size=size)
        features = featurizer.featurize(self.complex_files)
        assert features.shape == (1, 3 * size)

    def test_splif_voxels_shape(self):
        box_width = 48
        voxel_width = 2
        voxels_per_edge = int(box_width / voxel_width)
        size = 8
        voxelizer = dc.feat.SplifVoxelizer(box_width=box_width,
                                           voxel_width=voxel_width,
                                           size=size)
        features = voxelizer.featurize(self.complex_files)
        assert features.shape == (1, voxels_per_edge, voxels_per_edge,
                                  voxels_per_edge, size * 3)
