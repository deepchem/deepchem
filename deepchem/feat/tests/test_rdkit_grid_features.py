"""
Test rdkit_grid_featurizer module.
"""
import os
import unittest

import numpy as np

from deepchem.feat import RdkitGridFeaturizer

np.random.seed(123)


class TestRdkitGridFeaturizer(unittest.TestCase):
    """
    Test RdkitGridFeaturizer class defined in rdkit_grid_featurizer module.
    """

    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        package_dir = os.path.dirname(os.path.dirname(current_dir))
        self.protein_file = os.path.join(package_dir, 'dock', 'tests',
                                         '1jld_protein.pdb')
        self.ligand_file = os.path.join(package_dir, 'dock', 'tests',
                                        '1jld_ligand.sdf')

    def test_default_featurizer(self):
        # test if default parameters work
        featurizer = RdkitGridFeaturizer()
        self.assertIsInstance(featurizer, RdkitGridFeaturizer)
        feature_tensor = featurizer.featurize([(self.ligand_file,
                                                self.protein_file)])
        self.assertIsInstance(feature_tensor, np.ndarray)

    def test_example_featurizer(self):
        # check if use-case from examples works
        featurizer = RdkitGridFeaturizer(
            voxel_width=1.0,
            box_width=75.0,
            feature_types=['ecfp', 'splif', 'hbond', 'salt_bridge'],
            ecfp_power=9,
            splif_power=9,
            flatten=True)
        feature_tensor = featurizer.featurize([(self.ligand_file,
                                                self.protein_file)])
        self.assertIsInstance(feature_tensor, np.ndarray)

    def test_force_flatten(self):
        # test if input is flattened when flat features are used
        featurizer = RdkitGridFeaturizer(feature_types=['ecfp_hashed'],
                                         flatten=False)
        featurizer.flatten = True  # False should be ignored with ecfp_hashed
        feature_tensor = featurizer.featurize([(self.ligand_file,
                                                self.protein_file)])
        self.assertIsInstance(feature_tensor, np.ndarray)
        self.assertEqual(feature_tensor.shape,
                         (1, 2 * 2**featurizer.ecfp_power))

    def test_combined(self):
        ecfp_power = 5
        splif_power = 5
        box_width = 75.0
        voxel_width = 1.0
        voxels_per_edge = int(box_width / voxel_width)

        # test voxel features
        featurizer = RdkitGridFeaturizer(voxel_width=voxel_width,
                                         box_width=box_width,
                                         feature_types=['voxel_combined'],
                                         ecfp_power=ecfp_power,
                                         splif_power=splif_power,
                                         flatten=False,
                                         sanitize=True)
        feature_tensor = featurizer.featurize([(self.ligand_file,
                                                self.protein_file)])
        self.assertIsInstance(feature_tensor, np.ndarray)
        voxel_total_len = (
            2**ecfp_power +
            len(featurizer.cutoffs['splif_contact_bins']) * 2**splif_power +
            len(featurizer.cutoffs['hbond_dist_bins']) + 5)
        self.assertEqual(feature_tensor.shape,
                         (1, voxels_per_edge, voxels_per_edge, voxels_per_edge,
                          voxel_total_len))

        # test flat features
        featurizer = RdkitGridFeaturizer(voxel_width=1.0,
                                         box_width=75.0,
                                         feature_types=['flat_combined'],
                                         ecfp_power=ecfp_power,
                                         splif_power=splif_power,
                                         sanitize=True)
        feature_tensor = featurizer.featurize([(self.ligand_file,
                                                self.protein_file)])
        self.assertIsInstance(feature_tensor, np.ndarray)
        flat_total_len = (
            3 * 2**ecfp_power +
            len(featurizer.cutoffs['splif_contact_bins']) * 2**splif_power +
            len(featurizer.cutoffs['hbond_dist_bins']))
        self.assertEqual(feature_tensor.shape, (1, flat_total_len))

        # check if aromatic features are ignored if sanitize=False
        featurizer = RdkitGridFeaturizer(voxel_width=1.0,
                                         box_width=75.0,
                                         feature_types=['all_combined'],
                                         ecfp_power=ecfp_power,
                                         splif_power=splif_power,
                                         flatten=True,
                                         sanitize=False)

        self.assertTrue('pi_stack' not in featurizer.feature_types)
        self.assertTrue('cation_pi' not in featurizer.feature_types)
        feature_tensor = featurizer.featurize([(self.ligand_file,
                                                self.protein_file)])
        self.assertIsInstance(feature_tensor, np.ndarray)
        self.assertEqual(feature_tensor.shape, (1, 56109538))

    def test_custom_cutoffs(self):
        custom_cutoffs = {
            'hbond_dist_bins': [(2., 3.), (3., 3.5)],
            'hbond_angle_cutoffs': [5, 90],
            'splif_contact_bins': [(0, 3.5), (3.5, 6.0)],
            'ecfp_cutoff': 5.0,
            'sybyl_cutoff': 3.0,
            'salt_bridges_cutoff': 4.0,
            'pi_stack_dist_cutoff': 5.0,
            'pi_stack_angle_cutoff': 15.0,
            'cation_pi_dist_cutoff': 5.5,
            'cation_pi_angle_cutoff': 20.0,
        }
        rgf_featurizer = RdkitGridFeaturizer(**custom_cutoffs)
        self.assertEqual(rgf_featurizer.cutoffs, custom_cutoffs)

    def test_rotations(self):
        featurizer = RdkitGridFeaturizer(nb_rotations=3,
                                         box_width=75.,
                                         voxel_width=1.,
                                         feature_types=['voxel_combined'],
                                         flatten=False,
                                         sanitize=True)
        feature_tensors = featurizer.featurize([(self.ligand_file,
                                                 self.protein_file)])
        self.assertEqual(feature_tensors.shape, (1, 300, 75, 75, 40))

        featurizer = RdkitGridFeaturizer(nb_rotations=3,
                                         box_width=75.,
                                         voxel_width=1.,
                                         feature_types=['flat_combined'],
                                         flatten=True,
                                         sanitize=True)
        feature_tensors = featurizer.featurize([(self.ligand_file,
                                                 self.protein_file)])
        self.assertEqual(feature_tensors.shape, (1, 204))

    def test_failures(self):
        # test flattened voxel features
        featurizer = RdkitGridFeaturizer(nb_rotations=0,
                                         box_width=75.,
                                         voxel_width=1.,
                                         feature_types=['voxel_combined'],
                                         flatten=True,
                                         sanitize=True)

        features = featurizer.featurize([(self.ligand_file, self.protein_file),
                                         ('nan', 'nan')])
        self.assertEqual(features.shape, (2, 16875000))

        # test voxel features
        featurizer = RdkitGridFeaturizer(nb_rotations=0,
                                         box_width=75.,
                                         voxel_width=1.,
                                         feature_types=['voxel_combined'],
                                         flatten=False,
                                         sanitize=True)
        features = featurizer.featurize([(self.ligand_file, self.protein_file),
                                         ('nan', 'nan')])
        self.assertEqual(features.shape, (2, 75, 75, 75, 40))

        # test flat features
        featurizer = RdkitGridFeaturizer(nb_rotations=0,
                                         box_width=75.,
                                         voxel_width=1.,
                                         feature_types=['flat_combined'],
                                         flatten=True,
                                         sanitize=True)
        features = featurizer.featurize([(self.ligand_file, self.protein_file),
                                         ('nan', 'nan')])
        self.assertEqual(features.shape, (2, 51))

        # test rotations
        featurizer = RdkitGridFeaturizer(nb_rotations=5,
                                         box_width=75.,
                                         voxel_width=1.,
                                         feature_types=['flat_combined'],
                                         flatten=True,
                                         sanitize=True)
        features = featurizer.featurize([(self.ligand_file, self.protein_file),
                                         ('nan', 'nan')])
        self.assertEqual(features.shape, (2, 306))
