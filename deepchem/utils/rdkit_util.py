import logging

import os
import numpy as np
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile


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


def add_hydrogens_f(mol):
  molecule_file = str(tempfile.NamedTemporaryFile().name)
  Chem.MolToPDBFile(mol, molecule_file)
  try:
    fixer = PDBFixer(filename=molecule_file)
    fixer.addMissingHydrogens(7.4)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(molecule_file, 'w'))
  except ValueError as e:
    print(e)
    raise MoleculeLoadException(e)
  return Chem.MolFromPDBFile(str(molecule_file), sanitize=False, removeHs=False)


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

  if calc_charges:
    my_mol = add_hydrogens_f(my_mol)
    compute_charges(my_mol)
  elif add_hydrogens:
    my_mol = add_hydrogens_f(my_mol)
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
