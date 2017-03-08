import logging

import os
import numpy as np
import tempfile
import shutil
from rdkit import Chem
import networkx as nx
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


def pdbqt_file_hack_protein(mol, outfile):
  lines = [x.strip() for x in open(outfile).readlines()]
  out_lines = []
  for line in lines:
    if "ROOT" in line or "ENDROOT" in line or "TORSDOF" in line:
      out_lines.append("%s\n" % line)
      continue
    if not line.startswith("ATOM"):
      continue
    line = line[:66]
    atom_index = int(line.split()[1])
    atom = mol.GetAtoms()[atom_index - 1]
    line = "%s    +0.000 %s\n" % (line, atom.GetSymbol().ljust(2))
    out_lines.append(line)
  with open(outfile, 'w') as fout:
    for line in out_lines:
      fout.write(line)


def pdbqt_file_hack_ligand(mol, outfile):
  PdbqtLigandWriter(mol, outfile).convert()


def write_molecule(mol, outfile, is_protein=False):
  if ".pdbqt" in outfile:
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
    if is_protein:
      pdbqt_file_hack_protein(mol, outfile)
    else:
      pdbqt_file_hack_ligand(mol, outfile)
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


class PdbqtLigandWriter(object):
  def __init__(self, mol, outfile):
    self.mol = mol
    self.outfile = outfile

  def convert(self):
    self._create_pdb_map()
    self._mol_to_graph()
    self._get_rotatable_bonds()

    for bond in self.rotatable_bonds:
      self.graph.remove_edge(bond[0], bond[1])
    self.components = [x for x in nx.connected_components(self.graph)]
    self._create_component_map(self.components)

    self.used_partitions = set()
    self.lines = []
    root = max(enumerate(self.components), key=lambda x: len(x[1]))[0]
    self.lines.append("ROOT\n")
    for atom in self.components[root]:
      self.lines.append(self._writer_line_for_atom(atom))
    self.lines.append("ENDROOT\n")
    self.used_partitions.add(root)
    for bond in self.rotatable_bonds:
      valid, next_partition = self._valid_bond(bond, root)
      if not valid:
        continue
      self._dfs(next_partition, bond)
    self.lines.append("TORSDOF %s" % len(self.rotatable_bonds))
    with open(self.outfile, 'w') as fout:
      for line in self.lines:
        fout.write(line)

  def _dfs(self, current_partition, bond):
    if self._get_component_for_atom(bond[1]) != current_partition:
      bond = (bond[1], bond[0])
    self.used_partitions.add(self._get_component_for_atom(bond[0]))
    self.used_partitions.add(self._get_component_for_atom(bond[1]))
    self.lines.append("BRANCH %4s %4s\n" % (bond[0] + 1, bond[1] + 1))
    for atom in self.components[current_partition]:
      self.lines.append(self._writer_line_for_atom(atom))
    for b in self.rotatable_bonds:
      valid, next_partition = self._valid_bond(b, current_partition)
      if not valid:
        continue
      self._dfs(next_partition, b)
    self.lines.append("ENDBRANCH %4s %4s\n" % (bond[0] + 1, bond[1] + 1))

  def _get_component_for_atom(self, atom_number):
    return self.comp_map[atom_number]

  def _valid_bond(self, bond, current_partition):
    part1 = self.comp_map[bond[0]]
    part2 = self.comp_map[bond[1]]
    if part1 != current_partition and part2 != current_partition:
      return False, 0
    if part1 == current_partition:
      next_partition = part2
    else:
      next_partition = part1
    return not next_partition in self.used_partitions, next_partition

  def _writer_line_for_atom(self, atom_number):
    return self.pdb_map[atom_number]

  def _create_component_map(self, components):
    comp_map = {}
    for i in range(self.mol.GetNumAtoms()):
      for j in range(len(components)):
        if i in components[j]:
          comp_map[i] = j
          break
    self.comp_map = comp_map

  def _create_pdb_map(self):
    lines = [x.strip() for x in open(self.outfile).readlines()]
    lines = filter(lambda x: x.startswith("HETATM") or x.startswith("ATOM"), lines)
    lines = [x[:66] for x in lines]
    pdb_map = {}
    for line in lines:
      my_values = line.split()
      atom_number = int(my_values[1])
      atom_symbol = my_values[2]
      line = line.replace("HETATM", "ATOM  ")
      line = "%s    +0.000 %s\n" % (line, atom_symbol.ljust(2))
      pdb_map[atom_number - 1] = line
    self.pdb_map = pdb_map

  def _mol_to_graph(self):
    G = nx.Graph()
    num_atoms = self.mol.GetNumAtoms()
    G.add_nodes_from(range(num_atoms))
    for i in range(self.mol.GetNumBonds()):
      from_idx = self.mol.GetBonds()[i].GetBeginAtomIdx()
      to_idx = self.mol.GetBonds()[i].GetEndAtomIdx()
      G.add_edge(from_idx, to_idx)
    self.graph = G

  def _get_rotatable_bonds(self):
    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("
                                 "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="
                                 "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#"
                                 "*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])"
                                 "[CH3])]")
    from rdkit.Chem import rdmolops
    rdmolops.FastFindRings(self.mol)
    self.rotatable_bonds = self.mol.GetSubstructMatches(pattern)
