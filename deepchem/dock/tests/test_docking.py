"""
Tests for Docking
"""
import os
import platform
import unittest
import pytest
import logging
import numpy as np
import deepchem as dc
from deepchem.feat import ComplexFeaturizer
from deepchem.models import Model
from deepchem.dock.pose_generation import PoseGenerator

IS_WINDOWS = platform.system() == 'Windows'


class TestDocking(unittest.TestCase):
    """Does sanity checks on pose generation."""

    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.protein_file = os.path.join(current_dir, "1jld_protein.pdb")
        self.ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    @pytest.mark.slow
    def test_docker_init(self):
        """Test that Docker can be initialized."""
        vpg = dc.dock.VinaPoseGenerator()
        dc.dock.Docker(vpg)

    @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
    @pytest.mark.slow
    def test_docker_dock(self):
        """Test that Docker can dock."""
        # We provide no scoring model so the docker won't score
        vpg = dc.dock.VinaPoseGenerator()
        docker = dc.dock.Docker(vpg)
        docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                     exhaustiveness=1,
                                     num_modes=1,
                                     out_dir="/tmp")

        # Check only one output since num_modes==1
        assert len(list(docked_outputs)) == 1

    @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
    @pytest.mark.slow
    def test_docker_pose_generator_scores(self):
        """Test that Docker can get scores from pose_generator."""
        # We provide no scoring model so the docker won't score
        vpg = dc.dock.VinaPoseGenerator()
        docker = dc.dock.Docker(vpg)
        docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                     exhaustiveness=1,
                                     num_modes=1,
                                     out_dir="/tmp",
                                     use_pose_generator_scores=True)

        # Check only one output since num_modes==1
        docked_outputs = list(docked_outputs)
        assert len(docked_outputs) == 1
        assert len(docked_outputs[0]) == 2

    @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
    @pytest.mark.slow
    def test_docker_specified_pocket(self):
        """Test that Docker can dock into spec. pocket."""
        # Let's turn on logging since this test will run for a while
        logging.basicConfig(level=logging.INFO)
        vpg = dc.dock.VinaPoseGenerator()
        docker = dc.dock.Docker(vpg)
        docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                     centroid=(10, 10, 10),
                                     box_dims=(10, 10, 10),
                                     exhaustiveness=1,
                                     num_modes=1,
                                     out_dir="/tmp")

        # Check returned files exist
        assert len(list(docked_outputs)) == 1

    @unittest.skipIf(IS_WINDOWS, "vina is not supported in windows")
    @pytest.mark.slow
    def test_pocket_docker_dock(self):
        """Test that Docker can find pockets and dock dock."""
        # Let's turn on logging since this test will run for a while
        logging.basicConfig(level=logging.INFO)
        pocket_finder = dc.dock.ConvexHullPocketFinder()
        vpg = dc.dock.VinaPoseGenerator(pocket_finder=pocket_finder)
        docker = dc.dock.Docker(vpg)
        docked_outputs = docker.dock((self.protein_file, self.ligand_file),
                                     exhaustiveness=1,
                                     num_modes=1,
                                     num_pockets=1,
                                     out_dir="/tmp")

        # Check returned files exist
        assert len(list(docked_outputs)) == 1

    @pytest.mark.slow
    def test_scoring_model_and_featurizer(self):
        """Test that scoring model and featurizer are invoked correctly."""

        class DummyFeaturizer(ComplexFeaturizer):

            def featurize(self, complexes, *args, **kwargs):
                return np.zeros((len(complexes), 5))

        class DummyModel(Model):

            def predict(self, dataset, *args, **kwargs):
                return np.zeros(len(dataset))

        class DummyPoseGenerator(PoseGenerator):

            def generate_poses(self, *args, **kwargs):
                return [None]

        featurizer = DummyFeaturizer()
        scoring_model = DummyModel()
        pose_generator = DummyPoseGenerator()
        docker = dc.dock.Docker(pose_generator, featurizer, scoring_model)
        outputs = docker.dock(None)
        assert list(outputs) == [(None, np.array([0.]))]
