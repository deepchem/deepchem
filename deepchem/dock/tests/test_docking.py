"""
Tests for Docking 
"""
import os
import sys
import unittest
import pytest
import logging
import deepchem as dc
from deepchem.dock.binding_pocket import ConvexHullPocketFinder


class TestDocking(unittest.TestCase):
  """
  Does sanity checks on pose generation.
  """

  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    self.ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

  @pytest.mark.slow
  def test_docker_init(self):
    """Test that Docker can be initialized."""
    vpg = dc.dock.VinaPoseGenerator()
    docker = dc.dock.Docker(vpg)

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

  @pytest.mark.slow
  def test_docker_specified_pocket(self):
    """Test that Docker can dock into spec. pocket."""
    # Let's turn on logging since this test will run for a while
    logging.basicConfig(level=logging.INFO)
    vpg = dc.dock.VinaPoseGenerator()
    docker = dc.dock.Docker(vpg)
    docked_outputs = docker.dock(
        (self.protein_file, self.ligand_file), 
        centroid=(10, 10, 10),
        box_dims=(1, 1, 1),
        exhaustiveness=1,
        num_modes=1,
        out_dir="/tmp")

    # Check returned files exist
    assert len(list(docked_outputs)) == 1 

  @pytest.mark.slow
  def test_pocket_docker_dock(self):
    """Test that Docker can find pockets and dock dock."""
    # Let's turn on logging since this test will run for a while
    logging.basicConfig(level=logging.INFO)
    pocket_finder = ConvexHullPocketFinder()
    vpg = dc.dock.VinaPoseGenerator(pocket_finder=pocket_finder)
    docker = dc.dock.Docker(vpg)
    docked_outputs = docker.dock(
        (self.protein_file, self.ligand_file),
        exhaustiveness=1,
        num_modes=1,
        num_pockets=1,
        out_dir="/tmp")

    # Check returned files exist
    assert len(list(docked_outputs)) == 1
