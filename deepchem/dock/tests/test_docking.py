"""
Tests for Docking 
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import unittest
import os
from nose.plugins.attrib import attr
from nose.tools import nottest
import sys
import deepchem as dc


class TestDocking(unittest.TestCase):
  """
  Does sanity checks on pose generation. 
  """

  def test_vina_grid_rf_docker_init(self):
    """Test that VinaGridRFDocker can be initialized."""
    if sys.version_info >= (3, 0):
      return
    docker = dc.dock.VinaGridRFDocker(exhaustiveness=1, detect_pockets=False)

  def test_pocket_vina_grid_rf_docker_init(self):
    """Test that VinaGridRFDocker w/pockets can be initialized."""
    if sys.version_info >= (3, 0):
      return
    docker = dc.dock.VinaGridRFDocker(exhaustiveness=1, detect_pockets=True)

  @attr("slow")
  def test_vina_grid_dnn_docker_init(self):
    """Test that VinaGridDNNDocker can be initialized."""
    docker = dc.dock.VinaGridDNNDocker(exhaustiveness=1, detect_pockets=False)

  def test_pocket_vina_grid_dnn_docker_init(self):
    """Test that VinaGridDNNDocker can be initialized."""
    if sys.version_info >= (3, 0):
      return
    docker = dc.dock.VinaGridDNNDocker(exhaustiveness=1, detect_pockets=True)

  def test_vina_grid_rf_docker_dock(self):
    """Test that VinaGridRFDocker can dock."""
    if sys.version_info >= (3, 0):
      return

    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    docker = dc.dock.VinaGridRFDocker(exhaustiveness=1, detect_pockets=False)
    (score, (protein_docked, ligand_docked)) = docker.dock(protein_file,
                                                           ligand_file)

    # Check returned files exist
    assert score.shape == (1,)
    assert os.path.exists(protein_docked)
    assert os.path.exists(ligand_docked)

  def test_vina_grid_rf_docker_specified_pocket(self):
    """Test that VinaGridRFDocker can dock into spec. pocket."""
    if sys.version_info >= (3, 0):
      return

    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    docker = dc.dock.VinaGridRFDocker(exhaustiveness=1, detect_pockets=False)
    (score, (protein_docked, ligand_docked)) = docker.dock(
        protein_file,
        ligand_file,
        centroid=(10, 10, 10),
        box_dims=(1, 1, 1),
        dry_run=True)

    # Check returned files exist
    assert score.shape == (1,)

  @nottest
  def test_pocket_vina_grid_rf_docker_dock(self):
    """Test that VinaGridRFDocker can dock."""
    if sys.version_info >= (3, 0):
      return

    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    docker = dc.dock.VinaGridRFDocker(exhaustiveness=1, detect_pockets=True)
    (score, (protein_docked, ligand_docked)) = docker.dock(
        protein_file, ligand_file, dry_run=True)

    # Check returned files exist
    if sys.version_info >= (3, 0):
      return

    assert score.shape == (1,)

  def test_vina_grid_dnn_docker_dock(self):
    """Test that VinaGridDNNDocker can dock."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    docker = dc.dock.VinaGridDNNDocker(exhaustiveness=1, detect_pockets=False)
    (score, (protein_docked, ligand_docked)) = docker.dock(protein_file,
                                                           ligand_file)

    # Check returned files exist
    assert score.shape == (1,)
    assert os.path.exists(protein_docked)
    assert os.path.exists(ligand_docked)

  @attr('slow')
  def test_pocket_vina_grid_dnn_docker_dock(self):
    """Test that VinaGridDNNDocker can dock."""
    if sys.version_info >= (3, 0):
      return

    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    docker = dc.dock.VinaGridDNNDocker(exhaustiveness=1, detect_pockets=True)
    (score, (protein_docked, ligand_docked)) = docker.dock(
        protein_file, ligand_file, dry_run=True)

    # Check returned files exist
    assert score.shape == (1,)
