"""
Computes putative binding pockets on protein.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "GPL"

import numpy as np
import os
import pybel
import tempfile
from scipy.spatial import ConvexHull
from deepchem.feat import hydrogenate_and_compute_partial_charges
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.grid_featurizer import load_molecule
from subprocess import call

def extract_active_site(protein_file, ligand_file, cutoff=4):
  """Extracts a box for the active site."""
  protein_coords = load_molecule(protein_file)[0]
  ligand_coords = load_molecule(ligand_file)[0]
  num_ligand_atoms = len(ligand_coords)
  num_protein_atoms = len(protein_coords)
  pocket_inds = []
  pocket_atoms = set([])
  for lig_atom_ind in range(num_ligand_atoms):
    lig_atom = ligand_coords[lig_atom_ind]
    for protein_atom_ind in range(num_protein_atoms):
      protein_atom = protein_coords[protein_atom_ind]
      if np.linalg.norm(lig_atom - protein_atom) < cutoff:
        if protein_atom_ind not in pocket_atoms:
          pocket_atoms = pocket_atoms.union(set([protein_atom_ind]))
  # Should be an array of size (n_pocket_atoms, 3)
  pocket_atoms = list(pocket_atoms)
  n_pocket_atoms = len(pocket_atoms)
  pocket_coords = np.zeros((n_pocket_atoms, 3))
  for ind, pocket_ind in enumerate(pocket_atoms):
    pocket_coords[ind] = protein_coords[pocket_ind]

  x_min = int(np.floor(np.amin(pocket_coords[:, 0])))
  x_max = int(np.ceil(np.amax(pocket_coords[:, 0])))
  y_min = int(np.floor(np.amin(pocket_coords[:, 1])))
  y_max = int(np.ceil(np.amax(pocket_coords[:, 1])))
  z_min = int(np.floor(np.amin(pocket_coords[:, 2])))
  z_max = int(np.ceil(np.amax(pocket_coords[:, 2])))
  return (((x_min, x_max), (y_min, y_max), (z_min, z_max)),
          pocket_atoms, pocket_coords)

def compute_overlap(mapping, box1, box2):
  """Computes overlap between the two boxes.

  Overlap is defined as % atoms of box1 in box2. Note that
  overlap is not a symmetric measurement.
  """
  atom1 = set(mapping[box1])
  atom2 = set(mapping[box2])
  return len(atom1.intersection(atom2))/float(len(atom1))

def get_all_boxes(coords, pad=5):
  """Get all pocket boxes for protein coords.

  We pad all boxes the prescribed number of angstroms.

  TODO(rbharath): It looks like this may perhaps be non-deterministic?
  """
  hull = ConvexHull(coords)
  boxes = []
  for triangle in hull.simplices:
    # coords[triangle, 0] gives the x-dimension of all triangle points
    # Take transpose to make sure rows correspond to atoms.
    points = np.array(
        [coords[triangle, 0], coords[triangle, 1], coords[triangle, 2]]).T
    # We voxelize so all grids have integral coordinates (convenience)
    x_min, x_max = np.amin(points[:, 0]), np.amax(points[:, 0])
    x_min, x_max = int(np.floor(x_min))-pad, int(np.ceil(x_max))+pad
    y_min, y_max = np.amin(points[:, 1]), np.amax(points[:, 1])
    y_min, y_max = int(np.floor(y_min))-pad, int(np.ceil(y_max))+pad
    z_min, z_max = np.amin(points[:, 2]), np.amax(points[:, 2])
    z_min, z_max = int(np.floor(z_min))-pad, int(np.ceil(z_max))+pad
    boxes.append(((x_min, x_max), (y_min, y_max), (z_min, z_max)))
  return boxes 

def boxes_to_atoms(atom_coords, boxes):
  """Maps each box to a list of atoms in that box.

  TODO(rbharath): This does a num_atoms x num_boxes computations. Is
  there a reasonable heuristic we can use to speed this up?
  """
  mapping = {}
  for box_ind, box in enumerate(boxes):
    box_atoms = []
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = box
    print("Handing box %d/%d" % (box_ind, len(boxes)))
    for atom_ind in range(len(atom_coords)):
      atom = atom_coords[atom_ind] 
      x_cont = x_min <= atom[0] and atom[0] <= x_max
      y_cont = y_min <= atom[1] and atom[1] <= y_max
      z_cont = z_min <= atom[2] and atom[2] <= z_max
      if x_cont and y_cont and z_cont:
        box_atoms.append(atom_ind)
    mapping[box] = box_atoms
  return mapping

def merge_boxes(box1, box2):
  """Merges two boxes."""
  (x_min1, x_max1), (y_min1, y_max1), (z_min1, z_max1) = box1 
  (x_min2, x_max2), (y_min2, y_max2), (z_min2, z_max2) = box2 
  x_min = min(x_min1, x_min2)
  y_min = min(y_min1, y_min2)
  z_min = min(z_min1, z_min2)
  x_max = max(x_max1, x_max2)
  y_max = max(y_max1, y_max2)
  z_max = max(z_max1, z_max2)
  return ((x_min, x_max), (y_min, y_max), (z_min, z_max))

def merge_overlapping_boxes(mapping, boxes, threshold=.8):
  """Merge boxes which have an overlap greater than threshold.

  TODO(rbharath): This merge code is terribly inelegant. It's also quadratic
  in number of boxes. It feels like there ought to be an elegant divide and
  conquer approach here. Figure out later...
  """
  num_boxes = len(boxes)
  outputs = []
  for i in range(num_boxes):
    box = boxes[0]
    new_boxes = []
    new_mapping = {}
    # If overlap of box with previously generated output boxes, return
    contained = False
    for output_box in outputs:
      # Carry forward mappings
      new_mapping[output_box] = mapping[output_box]
      if compute_overlap(mapping, box, output_box) == 1:
        contained = True
    if contained:
      continue
    # We know that box has at least one atom not in outputs
    unique_box = True
    for merge_box in boxes[1:]:
      overlap = compute_overlap(mapping, box, merge_box)
      if overlap < threshold:
        new_boxes.append(merge_box)
        new_mapping[merge_box] = mapping[merge_box]
      else:
        # Current box has been merged into box further down list.
        # No need to output current box
        unique_box = False
        merged = merge_boxes(box, merge_box)
        new_boxes.append(merged)
        new_mapping[merged] = list(
            set(mapping[box]).union(set(mapping[merge_box])))
    if unique_box:
      outputs.append(box)
      new_mapping[box] = mapping[box]
    boxes = new_boxes
    mapping = new_mapping
  return outputs, mapping

class BindingPocketFinder(object):
  """Abstract superclass for binding pocket detectors"""

  def find_pockets(self, protein_file):
    """Finds potential binding pockets in proteins."""
    raise NotImplementedError

class ConvexHullPocketFinder(BindingPocketFinder):
  """Implementation that uses convex hull of protein to find pockets.

  Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4112621/pdf/1472-6807-14-18.pdf
  """
  def __init__(self, pad=5):
    self.pad = pad

  def find_all_pockets(self, protein_file):
    """Find list of binding pockets on protein."""
    # protein_coords is (N, 3) tensor
    coords = load_molecule(protein_file)[0]
    return get_all_boxes(coords, self.pad)

  def find_pockets(self, protein_file, ligand_file):
    """Find list of suitable binding pockets on protein."""
    protein_coords = load_molecule(protein_file)[0]
    ligand_coords = load_molecule(ligand_file)[0]
    boxes = get_all_boxes(protein_coords, self.pad)
    mapping = boxes_to_atoms(protein_coords, boxes)
    merged_boxes, mapping = merge_overlapping_boxes(mapping, boxes)
    return merged_boxes, mapping
