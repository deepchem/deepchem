import os
import unittest
import deepchem as dc


class TestContactFeaturizers(unittest.TestCase):
    """Test Contact Fingerprints and Voxelizers."""

    def setUp(self):
        # TODO test more formats for ligand
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.protein_file = os.path.join(current_dir, 'data',
                                         '3ws9_protein_fixer_rdkit.pdb')
        self.ligand_file = os.path.join(current_dir, 'data', '3ws9_ligand.sdf')
        self.complex_files = [(self.ligand_file, self.protein_file)]

    def test_contact_fingerprint_shape(self):
        size = 8
        featurizer = dc.feat.ContactCircularFingerprint(size=size)
        features = featurizer.featurize(self.complex_files)
        assert features.shape == (1, 2 * size)

    def test_contact_voxels_shape(self):
        box_width = 48
        voxel_width = 2
        voxels_per_edge = box_width / voxel_width
        size = 8
        voxelizer = dc.feat.ContactCircularVoxelizer(box_width=box_width,
                                                     voxel_width=voxel_width,
                                                     size=size)
        features = voxelizer.featurize(self.complex_files)
        assert features.shape == (1, voxels_per_edge, voxels_per_edge,
                                  voxels_per_edge, size)

    def test_contact_voxels_flattened(self):
        box_width = 48
        voxel_width = 2
        voxels_per_edge = box_width / voxel_width
        size = 8
        voxelizer = dc.feat.ContactCircularVoxelizer(box_width=box_width,
                                                     voxel_width=voxel_width,
                                                     size=size,
                                                     flatten=True)
        features = voxelizer.featurize(self.complex_files)
        assert features.shape == (1, int(size * voxels_per_edge**3))
