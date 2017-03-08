import logging

import os
import numpy as np
import tempfile
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile

try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO


class MoleculeLoadException(Exception):

  def __init__(self, *args, **kwargs):
    Exception.__init__(*args, **kwargs)


def get_xyz_from_mol(mol):
  """
  returns an m x 3 np array of 3d coords
  of given rdkit molecule
  """
  xyz = np.zeros((mol.GetNumAtoms(), 3))
  conf = mol.GetConformer()
  for i in range(conf.GetNumAtoms()):
    position = conf.GetAtomPosition(i)
    xyz[i, 0] = position.x
    xyz[i, 1] = position.y
    xyz[i, 2] = position.z
  return (xyz)


def add_hydrogens_to_mol(mol):
  molecule_file = None
  try:
    pdbblock = Chem.MolToPDBBlock(mol)
    pdb_stringio = StringIO()
    pdb_stringio.write(pdbblock)
    pdb_stringio.seek(0)
    fixer = PDBFixer(pdbfile=pdb_stringio)
    fixer.addMissingHydrogens(7.4)

    hydrogenated_io = StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, hydrogenated_io)
    hydrogenated_io.seek(0)
    return Chem.MolFromPDBBlock(
        hydrogenated_io.read(), sanitize=False, removeHs=False)
  except ValueError as e:
    logging.warning("Unable to add hydrogens", e)
    raise MoleculeLoadException(e)
  finally:
    try:
      os.remove(molecule_file)
    except (OSError, TypeError):
      pass


def compute_charges(mol):
  try:
    AllChem.ComputeGasteigerCharges(mol)
  except Exception as e:
    logging.exception("Unable to compute charges for mol")
    raise MoleculeLoadException(e)
  return mol


def load_molecule(molecule_file, add_hydrogens=True, calc_charges=True):
  """Converts molecule file to (xyz-coords, obmol object)

  Given molecule_file, returns a tuple of xyz coords of molecule
  and an rdkit object representing that molecule
  """
  if ".mol2" in molecule_file or ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
    my_mol = suppl[0]
  elif ".pdbqt" in molecule_file:
    raise MoleculeLoadException("Don't support pdbqt files yet")
  elif ".pdb" in molecule_file:
    my_mol = Chem.MolFromPDBFile(
        str(molecule_file), sanitize=False, removeHs=False)
  else:
    raise ValueError("Unrecognized file type")

  if my_mol is None:
    raise ValueError("Unable to read non None Molecule Object")

  if add_hydrogens:
    my_mol = add_hydrogens_to_mol(my_mol)
  if calc_charges:
    compute_charges(my_mol)

  xyz = get_xyz_from_mol(my_mol)

  return xyz, my_mol


def write_molecule(mol, outfile):
  if ".pdbqt" in outfile:
    # TODO (LESWING) create writer for pdbqt which includes charges
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
    pass
  elif ".pdb" in outfile:
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
  else:
    raise ValueError("Unsupported Format")


def pdbqt_to_pdb(filename):
  base_filename = os.path.splitext(filename)[0]
  pdb_filename = base_filename + ".pdb"
  pdbqt_data = open(filename).readlines()
  with open(pdb_filename, 'w') as fout:
    for line in pdbqt_data:
      fout.write("%s\n" % line[:66])
  return pdb_filename
