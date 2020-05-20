"""
Computes putative binding pockets on protein.
"""
import os
import logging
import tempfile
import numpy as np
from subprocess import call
from deepchem.feat.fingerprints import CircularFingerprint
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils import rdkit_util
from deepchem.utils import coordinate_box_utils as box_utils
from deepchem.utils.fragment_util import get_contact_atom_indices

logger = logging.getLogger(__name__)

def extract_active_site(protein_file, ligand_file, cutoff=4):
  """Extracts a box for the active site.

  Params
  ------
  protein_file: str
    Location of protein PDB
  ligand_file: str
    Location of ligand input file
  cutoff: int, optional
    The distance in angstroms from the protein pocket to
    consider for featurization.
  """
  protein = rdkit_util.load_molecule(
      protein_file, add_hydrogens=False)
  ligand = rdkit_util.load_molecule(
      ligand_file, add_hydrogens=True, calc_charges=True)
  protein_contacts, ligand_contacts = get_contact_atom_indices([protein, ligand], cutoff=cutoff)
  protein_coords = protein[0]
  pocket_coords = protein_coords[protein_contacts]

  x_min = int(np.floor(np.amin(pocket_coords[:, 0])))
  x_max = int(np.ceil(np.amax(pocket_coords[:, 0])))
  y_min = int(np.floor(np.amin(pocket_coords[:, 1])))
  y_max = int(np.ceil(np.amax(pocket_coords[:, 1])))
  z_min = int(np.floor(np.amin(pocket_coords[:, 2])))
  z_max = int(np.ceil(np.amax(pocket_coords[:, 2])))
  box = box_utils.CoordinateBox((x_min, x_max), (y_min, y_max), (z_min, z_max))
  return (box, pocket_coords)

class BindingPocketFinder(object):
  """Abstract superclass for binding pocket detectors

  Many times when working with a new protein or other macromolecule,
  it's not clear what zones of the macromolecule may be good targets
  for potential ligands or other molecules to interact with. This
  abstract class provides a template for child classes that
  algorithmically locate potential binding pockets that are good
  potential interaction sites.

  Note that potential interactions sites can be found by many
  different methods, and that this abstract class doesn't specify the
  technique to be used.
  """

  def find_pockets(self, molecule):
    """Finds potential binding pockets in proteins.

    Parameters
    ----------
    molecule: object
      Some representation of a molecule.
    """
    raise NotImplementedError


class ConvexHullPocketFinder(BindingPocketFinder):
  """Implementation that uses convex hull of protein to find pockets.

  Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4112621/pdf/1472-6807-14-18.pdf
  """

  def __init__(self, scoring_model=None, pad=5):
    """Initialize the pocket finder.

    Parameters
    ----------
    scoring_model: `dc.models.Model`, optional
      If specified, use this model to prune pockets.
    pad: float, optional
      The number of angstroms to pad around a binding pocket's atoms
      to get a binding pocket box.
    """
    self.scoring_model = scoring_model
    self.pad = pad

  def find_all_pockets(self, protein_file):
    """Find list of binding pockets on protein.
    
    Parameters
    ----------
    protein_file: str
      Protein to load in.
    """
    coords, _ = rdkit_util.load_molecule(protein_file)
    return box_utils.get_face_boxes(coords, self.pad)

  def find_pockets(self, macromolecule_file):
    """Find list of suitable binding pockets on protein.

    This function computes putative binding pockets on this protein.
    This class uses the `ConvexHull` to compute binding pockets. Each
    face of the hull is converted into a coordinate box used for
    binding.

    Params
    ------
    macromolecule_file: str
      Location of the macromolecule file to load

    Returns
    -------
    List of pockets. Each pocket is a `CoordinateBox`
    """
    coords = rdkit_util.load_molecule(
        macromolecule_file, add_hydrogens=False, calc_charges=False)[0]
    boxes = box_utils.get_face_boxes(coords, self.pad)
    boxes = box_utils.merge_overlapping_boxes(boxes)
    return boxes
