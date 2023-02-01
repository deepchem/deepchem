"""
Tests for binding pocket detection.
"""
import os
import logging
import unittest
import numpy as np

import deepchem as dc
from deepchem.utils import rdkit_utils
from deepchem.utils import coordinate_box_utils as box_utils

logger = logging.getLogger(__name__)


class TestBindingPocket(unittest.TestCase):
    """Does sanity checks on binding pocket generation."""

    def test_convex_init(self):
        """Tests that ConvexHullPocketFinder can be initialized."""
        dc.dock.ConvexHullPocketFinder()

    def test_get_face_boxes_for_protein(self):
        """Tests that binding pockets are detected."""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        protein_file = os.path.join(current_dir, "1jld_protein.pdb")
        coords = rdkit_utils.load_molecule(protein_file)[0]

        boxes = box_utils.get_face_boxes(coords)
        assert isinstance(boxes, list)
        # Pocket is of form ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        for pocket in boxes:
            assert isinstance(pocket, box_utils.CoordinateBox)

    def test_convex_find_pockets(self):
        """Test that some pockets are filtered out."""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        protein_file = os.path.join(current_dir, "1jld_protein.pdb")

        finder = dc.dock.ConvexHullPocketFinder()
        all_pockets = finder.find_all_pockets(protein_file)
        pockets = finder.find_pockets(protein_file)
        # Test that every atom in pocket maps exists
        for pocket in pockets:
            assert isinstance(pocket, box_utils.CoordinateBox)

        assert len(pockets) < len(all_pockets)

    def test_extract_active_site(self):
        """Test that computed pockets have strong overlap with true binding pocket."""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        protein_file = os.path.join(current_dir, "1jld_protein.pdb")
        ligand_file = os.path.join(current_dir, "1jld_ligand.sdf")

        active_site_box, active_site_coords = \
          dc.dock.binding_pocket.extract_active_site(protein_file, ligand_file)

        assert isinstance(active_site_box, box_utils.CoordinateBox)
        assert isinstance(active_site_coords, np.ndarray)
