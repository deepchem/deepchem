"""
Tests for Pose Generation 
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import sys
import unittest
import deepchem as dc
from nose.plugins.attrib import attr


class TestPoseGeneration(unittest.TestCase):
  """
  Does sanity checks on pose generation. 
  """

  def test_vina_initialization(self):
    """Test that VinaPoseGenerator can be initialized."""
    # Note this may download autodock Vina...
    vpg = dc.dock.VinaPoseGenerator(detect_pockets=False, exhaustiveness=1)

  def test_pocket_vina_initialization(self):
    """Test that VinaPoseGenerator can be initialized."""
    # Note this may download autodock Vina...
    if sys.version_info >= (3, 0):
      return
    vpg = dc.dock.VinaPoseGenerator(detect_pockets=True, exhaustiveness=1)

  def test_vina_poses(self):
    """Test that VinaPoseGenerator creates pose files."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    # Note this may download autodock Vina...
    vpg = dc.dock.VinaPoseGenerator(detect_pockets=False, exhaustiveness=1)
    protein_pose_file, ligand_pose_file = vpg.generate_poses(
        protein_file, ligand_file, out_dir="/tmp")

    # Check returned files exist
    assert os.path.exists(protein_pose_file)
    assert os.path.exists(ligand_pose_file)

  @attr('slow')
  def test_pocket_vina_poses(self):
    """Test that VinaPoseGenerator creates pose files."""
    if sys.version_info >= (3, 0):
      return
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    # Note this may download autodock Vina...
    vpg = dc.dock.VinaPoseGenerator(detect_pockets=True, exhaustiveness=1)
    protein_pose_file, ligand_pose_file = vpg.generate_poses(
        protein_file, ligand_file, out_dir="/tmp")

    # Check returned files exist
    assert os.path.exists(protein_pose_file)
    assert os.path.exists(ligand_pose_file)
