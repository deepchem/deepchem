"""
Featurizes proposed binding pockets.
"""
import numpy as np
import logging
from deepchem.utils import rdkit_util
from deepchem.feat import Featurizer

logger = logging.getLogger(__name__)


def boxes_to_atoms(coords, boxes):
  """Maps each box to a list of atoms in that box.

  Given the coordinates of a macromolecule, and a collection of boxes,
  returns a dictionary which maps boxes to the atom indices of the
  atoms in them.

  Parameters
  ----------
  coords: np.ndarray
    Of shape `(N, 3)
  boxes: list
    list of `CoordinateBox` objects.

  Returns
  -------
  dictionary mapping `CoordinateBox` objects to lists of atom coordinates
  """
  mapping = {}
  for box_ind, box in enumerate(boxes):
    box_atoms = []
    for atom_ind in range(len(coords)):
      atom = coords[atom_ind]
      if atom in box:
        box_atoms.append(atom_ind)
    mapping[box] = box_atoms
  return mapping


class BindingPocketFeaturizer(Featurizer):
  """Featurizes binding pockets with information about chemical
  environments.

  In many applications, it's desirable to look at binding pockets on
  macromolecules which may be good targets for potential ligands or
  other molecules to interact with. A `BindingPocketFeaturizer`
  expects to be given a macromolecule, and a list of pockets to
  featurize on that macromolecule. These pockets should be of the form
  produced by a `dc.dock.BindingPocketFinder`, that is as a list of
  `dc.utils.CoordinateBox` objects.

  The base featurization in this class's featurization is currently
  very simple and counts the number of residues of each type present
  in the pocket. It's likely that you'll want to overwrite this
  implementation for more sophisticated downstream usecases. Note that
  this class's implementation will only work for proteins and not for
  other macromolecules
  """

  residues = [
      "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "PYL", "SER", "SEC", "THR", "TRP",
      "TYR", "VAL", "ASX", "GLX"
  ]

  n_features = len(residues)

  def featurize(self, protein_file, pockets):
    """
    Calculate atomic coodinates.

    Params
    ------
    protein_file: str
      Location of PDB file. Will be loaded by MDTraj
    pockets: list[CoordinateBox]
      List of `dc.utils.CoordinateBox` objects.

    Returns
    -------
    A numpy array of shale `(len(pockets), n_residues)`
    """
    import mdtraj
    protein_coords = rdkit_util.load_molecule(
        protein_file, add_hydrogens=False, calc_charges=False)[0]
    mapping = boxes_to_atoms(protein_coords, pockets)
    protein = mdtraj.load(protein_file)
    n_pockets = len(pockets)
    n_residues = len(BindingPocketFeaturizer.residues)
    res_map = dict(zip(BindingPocketFeaturizer.residues, range(n_residues)))
    all_features = np.zeros((n_pockets, n_residues))
    for pocket_num, pocket in enumerate(pockets):
      pocket_atoms = mapping[pocket]
      for ind, atom in enumerate(pocket_atoms):
        atom_name = str(protein.top.atom(atom))
        # atom_name is of format RESX-ATOMTYPE
        # where X is a 1 to 4 digit number
        residue = atom_name[:3]
        if residue not in res_map:
          logger.info("Warning: Non-standard residue in PDB file")
          continue
        atomtype = atom_name.split("-")[1]
        all_features[pocket_num, res_map[residue]] += 1
    return all_features
