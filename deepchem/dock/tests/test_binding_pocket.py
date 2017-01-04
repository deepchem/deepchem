"""
Tests for Pose Generation 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import unittest
import tempfile
import os
import shutil
import numpy as np
import deepchem as dc

class TestPoseGeneration(unittest.TestCase):
  """
  Does sanity checks on pose generation. 
  """

  def test_convex_rf_init(self):
    """Tests that ConvexHullPocketFinder can be initialized."""
    finder = dc.dock.ConvexHullPocketFinder()

  def test_get_all_boxes(self):
    """Tests that binding pockets are detected."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")
    coords = dc.feat.grid_featurizer.load_molecule(protein_file)[0]

    boxes = dc.dock.binding_pocket.get_all_boxes(coords)
    assert isinstance(boxes, list)
    # Pocket is of form ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    for pocket in boxes:
      assert len(pocket) == 3
      assert len(pocket[0]) == 2
      assert len(pocket[1]) == 2
      assert len(pocket[2]) == 2
      (x_min, x_max), (y_min, y_max), (z_min, z_max) = pocket
      assert x_min < x_max
      assert y_min < y_max
      assert z_min < z_max

  def test_boxes_to_atoms(self):
    """Test that mapping of protein atoms to boxes is meaningful."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")
    coords = dc.feat.grid_featurizer.load_molecule(protein_file)[0]
    boxes = dc.dock.binding_pocket.get_all_boxes(coords)

    mapping = dc.dock.binding_pocket.boxes_to_atoms(coords, boxes)
    assert isinstance(mapping, dict)
    for box, box_atoms in mapping.iteritems():
      (x_min, x_max), (y_min, y_max), (z_min, z_max) = box
      for atom_ind in box_atoms:
        atom = coords[atom_ind]
        assert x_min <= atom[0] and atom[0] <= x_max
        assert y_min <= atom[1] and atom[1] <= y_max
        assert z_min <= atom[2] and atom[2] <= z_max

  def test_compute_overlap(self):
    """Tests that overlap between boxes is computed correctly."""
    # box1 contained in box2
    box1 = ((1, 2), (1, 2), (1, 2))
    box2 = ((1, 3), (1, 3), (1, 3))
    mapping = {box1: [1, 2, 3, 4], box2: [1, 2, 3, 4, 5]}
    # box1 in box2, so complete overlap
    np.testing.assert_almost_equal(
        dc.dock.binding_pocket.compute_overlap(mapping, box1, box2), 1)
    # 4/5 atoms in box2 in box1, so 80 % overlap
    np.testing.assert_almost_equal(
        dc.dock.binding_pocket.compute_overlap(mapping, box2, box1), .8)

  def test_merge_overlapping_boxes(self):
    """Tests that overlapping boxes are merged."""
    # box2 contains box1
    box1 = ((1, 2), (1, 2), (1, 2))
    box2 = ((1, 3), (1, 3), (1, 3))
    mapping = {box1: [1, 2, 3, 4], box2: [1, 2, 3, 4, 5]}
    boxes = [box1, box2]
    merged_boxes, _ = dc.dock.binding_pocket.merge_overlapping_boxes(
        mapping, boxes)
    print("merged_boxes")
    print(merged_boxes)
    assert len(merged_boxes) == 1
    assert merged_boxes[0] == ((1, 3), (1, 3), (1, 3))

    # box1 contains box2
    box1 = ((1, 3), (1, 3), (1, 3))
    box2 = ((1, 2), (1, 2), (1, 2))
    mapping = {box1: [1, 2, 3, 4, 5, 6], box2: [1, 2, 3, 4]}
    boxes = [box1, box2]
    merged_boxes, _ = dc.dock.binding_pocket.merge_overlapping_boxes(
        mapping, boxes)
    print("merged_boxes")
    print(merged_boxes)
    assert len(merged_boxes) == 1
    assert merged_boxes[0] == ((1, 3), (1, 3), (1, 3))

    # box1 contains box2, box3
    box1 = ((1, 3), (1, 3), (1, 3))
    box2 = ((1, 2), (1, 2), (1, 2))
    box3 = ((1, 2.5), (1, 2.5), (1, 2.5))
    mapping = {box1: [1, 2, 3, 4, 5, 6], box2: [1, 2, 3, 4],
               box3: [1, 2, 3, 4, 5]}
    merged_boxes, _ = dc.dock.binding_pocket.merge_overlapping_boxes(
        mapping, boxes)
    print("merged_boxes")
    print(merged_boxes)
    assert len(merged_boxes) == 1
    assert merged_boxes[0] == ((1, 3), (1, 3), (1, 3))

  def test_convex_rf_find_pockets(self):
    """Test that some pockets are filtered out."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    finder = dc.dock.ConvexHullPocketFinder()

    all_pockets = finder.find_all_pockets(protein_file)
    pockets = finder.find_pockets(protein_file, ligand_file)

    assert len(pockets) < len(all_pockets)

  def test_extract_active_site(self):
    """Test that computed pockets have strong overlap with true binding pocket."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

    active_site_box, active_site_atoms, active_site_coords = (
        dc.dock.binding_pocket.extract_active_site(
            protein_file, ligand_file))
    print("active_site_box")
    print(active_site_box)
    print("len(active_site_atoms)")
    print(len(active_site_atoms))

    finder = dc.dock.ConvexHullPocketFinder()
    pockets, pocket_atoms = finder.find_pockets(protein_file, ligand_file)

    # Add active site to dict
    print("active_site_box")
    print(active_site_box)
    pocket_atoms[active_site_box] = active_site_atoms
    overlapping_pocket = False
    for pocket in pockets:
      print("pocket")
      print(pocket)
      overlap = dc.dock.binding_pocket.compute_overlap(
          pocket_atoms, active_site_box, pocket)
      if overlap > .5:
        overlapping_pocket = True
      print("Overlap for pocket is %f" % overlap)
    assert overlapping_pocket
    
