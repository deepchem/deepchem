"""
RDKit Utilities.

This file contains utilities that compute useful properties of
molecules. Some of these are simple cleanup utilities, and
others are more sophisticated functions that detect chemical
properties of molecules.
"""

import os
import logging
import itertools
import numpy as np
from io import StringIO
from copy import deepcopy
from collections import Counter
from deepchem.utils import pdbqt_utils
from deepchem.utils.pdbqt_utils import convert_mol_to_pdbqt
from deepchem.utils.pdbqt_utils import convert_protein_to_pdbqt
from deepchem.utils.geometry_utils import angle_between
from deepchem.utils.geometry_utils import is_angle_within_cutoff
from deepchem.utils.geometry_utils import generate_random_rotation_matrix

logger = logging.getLogger(__name__)


class MoleculeLoadException(Exception):

  def __init__(self, *args, **kwargs):
    Exception.__init__(*args, **kwargs)


def get_xyz_from_mol(mol):
  """Extracts a numpy array of coordinates from a molecules.

  Returns a `(N, 3)` numpy array of 3d coords of given rdkit molecule

  Parameters
  ----------
  mol: rdkit Molecule
    Molecule to extract coordinates for

  Returns
  -------
  Numpy ndarray of shape `(N, 3)` where `N = mol.GetNumAtoms()`.
  """
  xyz = np.zeros((mol.GetNumAtoms(), 3))
  conf = mol.GetConformer()
  for i in range(conf.GetNumAtoms()):
    position = conf.GetAtomPosition(i)
    xyz[i, 0] = position.x
    xyz[i, 1] = position.y
    xyz[i, 2] = position.z
  return (xyz)


def add_hydrogens_to_mol(mol, is_protein=False):
  """
  Add hydrogens to a molecule object

  Parameters
  ----------
  mol: Rdkit Mol
    Molecule to hydrogenate
  is_protein: bool, optional (default False)
    Whether this molecule is a protein.


  Returns
  -------
  Rdkit Mol

  Note
  ----
  This function requires RDKit and PDBFixer to be installed.
  """
  return apply_pdbfixer(mol, hydrogenate=True, is_protein=is_protein)


def apply_pdbfixer(mol,
                   add_missing=True,
                   hydrogenate=True,
                   pH=7.4,
                   remove_heterogens=True,
                   is_protein=True):
  """
  Apply PDBFixer to a molecule to try to clean it up.

  Parameters
  ----------
  mol: Rdkit Mol
    Molecule to clean up.
  add_missing: bool, optional
    If true, add in missing residues and atoms
  hydrogenate: bool, optional
    If true, add hydrogens at specified pH
  pH: float, optional
    The pH at which hydrogens will be added if `hydrogenate==True`. Set to 7.4 by default.
  remove_heterogens: bool, optional
    Often times, PDB files come with extra waters and salts attached.
    If this field is set, remove these heterogens.
  is_protein: bool, optional
    If false, then don't remove heterogens (since this molecule is
    itself a heterogen).
  
  Returns
  -------
  Rdkit Mol

  Note
  ----
  This function requires RDKit and PDBFixer to be installed.
  """
  molecule_file = None
  try:
    from rdkit import Chem
    pdbblock = Chem.MolToPDBBlock(mol)
    pdb_stringio = StringIO()
    pdb_stringio.write(pdbblock)
    pdb_stringio.seek(0)
    import pdbfixer
    fixer = pdbfixer.PDBFixer(pdbfile=pdb_stringio)
    if add_missing:
      fixer.findMissingResidues()
      fixer.findMissingAtoms()
      fixer.addMissingAtoms()
    if hydrogenate:
      fixer.addMissingHydrogens(pH)
    if is_protein and remove_heterogens:
      # False here specifies that water is to be removed
      fixer.removeHeterogens(False)

    hydrogenated_io = StringIO()
    import simtk
    simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions,
                                       hydrogenated_io)
    hydrogenated_io.seek(0)
    return Chem.MolFromPDBBlock(
        hydrogenated_io.read(), sanitize=False, removeHs=False)
  except ValueError as e:
    logger.warning("Unable to add hydrogens %s", e)
    raise MoleculeLoadException(e)
  finally:
    try:
      os.remove(molecule_file)
    except (OSError, TypeError):
      pass


def compute_charges(mol):
  """Attempt to compute Gasteiger Charges on Mol

  This also has the side effect of calculating charges on mol.  The
  mol passed into this function has to already have been sanitized

  Params
  ------
  mol: rdkit molecule

  Returns
  -------
  No return since updates in place.
  
  Note
  ----
  This function requires RDKit to be installed.
  """
  from rdkit.Chem import AllChem
  try:
    # Updates charges in place
    AllChem.ComputeGasteigerCharges(mol)
  except Exception as e:
    logging.exception("Unable to compute charges for mol")
    raise MoleculeLoadException(e)


def load_complex(molecular_complex,
                 add_hydrogens=True,
                 calc_charges=True,
                 sanitize=True):
  """Loads a molecular complex.

  Given some representation of a molecular complex, returns a list of
  tuples, where each tuple contains (xyz coords, rdkit object) for
  that constituent molecule in the complex.

  For now, assumes that molecular_complex is a tuple of filenames.

  Parameters
  ----------
  molecular_complex: list or str
    If list, each entry should be a filename for a constituent
    molecule in complex. If str, should be the filename of a file that
    holds the full complex.
  add_hydrogens: bool, optional
    If true, add hydrogens via pdbfixer
  calc_charges: bool, optional
    If true, add charges via rdkit
  sanitize: bool, optional
    If true, sanitize molecules via rdkit

  Returns
  -------
  List of tuples (xyz, mol)

  Note
  ----
  This function requires RDKit to be installed.
  """
  if isinstance(molecular_complex, str):
    molecule_complex = [molecular_complex]
  fragments = []
  for mol in molecular_complex:
    loaded = load_molecule(
        mol,
        add_hydrogens=add_hydrogens,
        calc_charges=calc_charges,
        sanitize=sanitize)
    if isinstance(loaded, list):
      fragments += loaded
    else:
      fragments.append(loaded)
  return fragments


def load_molecule(molecule_file,
                  add_hydrogens=True,
                  calc_charges=True,
                  sanitize=True):
  """Converts molecule file to (xyz-coords, obmol object)

  Given molecule_file, returns a tuple of xyz coords of molecule
  and an rdkit object representing that molecule in that order `(xyz,
  rdkit_mol)`. This ordering convention is used in the code in a few
  places.

  Parameters
  ----------
  molecule_file: str
    filename for molecule
  add_hydrogens: bool, optional
    If true, add hydrogens via pdbfixer
  calc_charges: bool, optional
    If true, add charges via rdkit
  sanitize: bool, optional
    If true, sanitize molecules via rdkit

  Returns
  -------
  Tuple (xyz, mol) if file contains single molecule. Else returns a
  list of the tuples for the separate molecules in this list.

  Note
  ----
  This function requires RDKit to be installed.
  """
  from rdkit import Chem
  from_pdb = False
  if ".mol2" in molecule_file:
    my_mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
  elif ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
    # TODO: This is wrong. Should return all molecules
    my_mol = suppl[0]
  elif ".pdbqt" in molecule_file:
    pdb_block = pdbqt_utils.pdbqt_to_pdb(molecule_file)
    my_mol = Chem.MolFromPDBBlock(
        str(pdb_block), sanitize=False, removeHs=False)
    from_pdb = True
  elif ".pdb" in molecule_file:
    my_mol = Chem.MolFromPDBFile(
        str(molecule_file), sanitize=False, removeHs=False)
    from_pdb = True
  else:
    raise ValueError("Unrecognized file type for %s" % str(molecule_file))

  if my_mol is None:
    raise ValueError("Unable to read non None Molecule Object")

  if add_hydrogens or calc_charges:
    # We assume if it's from a PDB, it should be a protein
    my_mol = apply_pdbfixer(
        my_mol, hydrogenate=add_hydrogens, is_protein=from_pdb)
  if sanitize:
    try:
      Chem.SanitizeMol(my_mol)
    # Ideally we should catch AtomValenceException but Travis seems to choke on it for some reason.
    except:
      logger.warn("Mol %s failed sanitization" % Chem.MolToSmiles(my_mol))
  if calc_charges:
    # This updates in place
    compute_charges(my_mol)

  xyz = get_xyz_from_mol(my_mol)

  return xyz, my_mol


def write_molecule(mol, outfile, is_protein=False):
  """Write molecule to a file

  This function writes a representation of the provided molecule to
  the specified `outfile`. Doesn't return anything.

  Parameters
  ----------
  mol: rdkit Mol
    Molecule to write
  outfile: str
    Filename to write mol to
  is_protein: bool, optional
    Is this molecule a protein?

  Note
  ----
  This function requires RDKit to be installed.

  Raises
  ------
  ValueError: if `outfile` isn't of a supported format.
  """
  from rdkit import Chem
  if ".pdbqt" in outfile:
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
    if is_protein:
      pdbqt_utils.convert_protein_to_pdbqt(mol, outfile)
    else:
      pdbqt_utils.convert_mol_to_pdbqt(mol, outfile)
  elif ".pdb" in outfile:
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
  elif ".sdf" in outfile:
    writer = Chem.SDWriter(outfile)
    writer.write(mol)
    writer.close()
  else:
    raise ValueError("Unsupported Format")


def merge_molecules_xyz(xyzs):
  """Merges coordinates of multiple molecules. 

  Parameters
  ----------
  xyzs: List
    List of numpy arrays each of shape `(N_i, 3)` where `N_i` is the number of atoms in the i-th atom.
  """
  return np.array(np.vstack(np.vstack(xyzs)))


def merge_molecules(molecules):
  """Helper method to merge two molecules.

  Parameters
  ----------
  molecules: list
    List of rdkit molecules

  Returns
  -------
  merged: rdkit molecule
  """
  from rdkit.Chem import rdmolops
  if len(molecules) == 0:
    return None
  elif len(molecules) == 1:
    return molecules[0]
  else:
    combined = molecules[0]
    for nextmol in molecules[1:]:
      combined = rdmolops.CombineMols(combined, nextmol)
    return combined
