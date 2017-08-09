"""
Computes putative binding pockets on protein.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import os
import tempfile
import numpy as np
from subprocess import call
from scipy.spatial import ConvexHull
from deepchem.feat.binding_pocket_features import BindingPocketFeaturizer
from deepchem.feat.fingerprints import CircularFingerprint
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils import rdkit_util


def extract_active_site(protein_file, ligand_file, cutoff=4):
  """Extracts a box for the active site."""
  protein_coords = rdkit_util.load_molecule(
      protein_file, add_hydrogens=False)[0]
  ligand_coords = rdkit_util.load_molecule(
      ligand_file, add_hydrogens=True, calc_charges=True)[0]
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
  return (((x_min, x_max), (y_min, y_max), (z_min, z_max)), pocket_atoms,
          pocket_coords)


def compute_overlap(mapping, box1, box2):
  """Computes overlap between the two boxes.

  Overlap is defined as % atoms of box1 in box2. Note that
  overlap is not a symmetric measurement.
  """
  atom1 = set(mapping[box1])
  atom2 = set(mapping[box2])
  return len(atom1.intersection(atom2)) / float(len(atom1))


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
    x_min, x_max = int(np.floor(x_min)) - pad, int(np.ceil(x_max)) + pad
    y_min, y_max = np.amin(points[:, 1]), np.amax(points[:, 1])
    y_min, y_max = int(np.floor(y_min)) - pad, int(np.ceil(y_max)) + pad
    z_min, z_max = np.amin(points[:, 2]), np.amax(points[:, 2])
    z_min, z_max = int(np.floor(z_min)) - pad, int(np.ceil(z_max)) + pad
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

  def find_pockets(self, protein_file, ligand_file):
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
    coords = rdkit_util.load_molecule(protein_file)[0]
    return get_all_boxes(coords, self.pad)

  def find_pockets(self, protein_file, ligand_file):
    """Find list of suitable binding pockets on protein."""
    protein_coords = rdkit_util.load_molecule(
        protein_file, add_hydrogens=False, calc_charges=False)[0]
    ligand_coords = rdkit_util.load_molecule(
        ligand_file, add_hydrogens=False, calc_charges=False)[0]
    boxes = get_all_boxes(protein_coords, self.pad)
    mapping = boxes_to_atoms(protein_coords, boxes)
    pockets, pocket_atoms_map = merge_overlapping_boxes(mapping, boxes)
    pocket_coords = []
    for pocket in pockets:
      atoms = pocket_atoms_map[pocket]
      coords = np.zeros((len(atoms), 3))
      for ind, atom in enumerate(atoms):
        coords[ind] = protein_coords[atom]
      pocket_coords.append(coords)
    return pockets, pocket_atoms_map, pocket_coords


class RFConvexHullPocketFinder(BindingPocketFinder):
  """Uses pre-trained RF model + ConvexHulPocketFinder to select pockets."""

  def __init__(self, pad=5):
    self.pad = pad
    self.convex_finder = ConvexHullPocketFinder(pad)

    # Load binding pocket model
    self.base_dir = tempfile.mkdtemp()
    print("About to download trained model.")
    # TODO(rbharath): Shift refined to full once trained.
    call((
        "wget -nv -c http://deepchem.io.s3-website-us-west-1.amazonaws.com/trained_models/pocket_random_refined_RF.tar.gz"
    ).split())
    call(("tar -zxvf pocket_random_refined_RF.tar.gz").split())
    call(("mv pocket_random_refined_RF %s" % (self.base_dir)).split())
    self.model_dir = os.path.join(self.base_dir, "pocket_random_refined_RF")

    # Fit model on dataset
    self.model = SklearnModel(model_dir=self.model_dir)
    self.model.reload()

    # Create featurizers
    self.pocket_featurizer = BindingPocketFeaturizer()
    self.ligand_featurizer = CircularFingerprint(size=1024)

  def find_pockets(self, protein_file, ligand_file):
    """Compute features for a given complex

    TODO(rbharath): This has a log of code overlap with
    compute_binding_pocket_features in
    examples/binding_pockets/binding_pocket_datasets.py. Find way to refactor
    to avoid code duplication.
    """
    # if not ligand_file.endswith(".sdf"):
    #   raise ValueError("Only .sdf ligand files can be featurized.")
    # ligand_basename = os.path.basename(ligand_file).split(".")[0]
    # ligand_mol2 = os.path.join(
    #     self.base_dir, ligand_basename + ".mol2")
    #
    # # Write mol2 file for ligand
    # obConversion = ob.OBConversion()
    # conv_out = obConversion.SetInAndOutFormats(str("sdf"), str("mol2"))
    # ob_mol = ob.OBMol()
    # obConversion.ReadFile(ob_mol, str(ligand_file))
    # obConversion.WriteFile(ob_mol, str(ligand_mol2))
    #
    # # Featurize ligand
    # mol = Chem.MolFromMol2File(str(ligand_mol2), removeHs=False)
    # if mol is None:
    #   return None, None
    # # Default for CircularFingerprint
    # n_ligand_features = 1024
    # ligand_features = self.ligand_featurizer.featurize([mol])
    #
    # # Featurize pocket
    # pockets, pocket_atoms_map, pocket_coords = self.convex_finder.find_pockets(
    #     protein_file, ligand_file)
    # n_pockets = len(pockets)
    # n_pocket_features = BindingPocketFeaturizer.n_features
    #
    # features = np.zeros((n_pockets, n_pocket_features+n_ligand_features))
    # pocket_features = self.pocket_featurizer.featurize(
    #     protein_file, pockets, pocket_atoms_map, pocket_coords)
    # # Note broadcast operation
    # features[:, :n_pocket_features] = pocket_features
    # features[:, n_pocket_features:] = ligand_features
    # dataset = NumpyDataset(X=features)
    # pocket_preds = self.model.predict(dataset)
    # pocket_pred_proba = np.squeeze(self.model.predict_proba(dataset))
    #
    # # Find pockets which are active
    # active_pockets = []
    # active_pocket_atoms_map = {}
    # active_pocket_coords = []
    # for pocket_ind in range(len(pockets)):
    #   #################################################### DEBUG
    #   # TODO(rbharath): For now, using a weak cutoff. Fix later.
    #   #if pocket_preds[pocket_ind] == 1:
    #   if pocket_pred_proba[pocket_ind][1] > .15:
    #   #################################################### DEBUG
    #     pocket = pockets[pocket_ind]
    #     active_pockets.append(pocket)
    #     active_pocket_atoms_map[pocket] = pocket_atoms_map[pocket]
    #     active_pocket_coords.append(pocket_coords[pocket_ind])
    # return active_pockets, active_pocket_atoms_map, active_pocket_coords
    # # TODO(LESWING)
    raise ValueError("Karl Implement")
