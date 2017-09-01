import logging

import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

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
  """
  Add hydrogens to a molecule object
  TODO (LESWING) see if there are more flags to add here for default
  :param mol: Rdkit Mol
  :return: Rdkit Mol
  """
  molecule_file = None
  try:
    pdbblock = Chem.MolToPDBBlock(mol)
    pdb_stringio = StringIO()
    pdb_stringio.write(pdbblock)
    pdb_stringio.seek(0)
    import pdbfixer
    fixer = pdbfixer.PDBFixer(pdbfile=pdb_stringio)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    hydrogenated_io = StringIO()
    import simtk
    simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions,
                                       hydrogenated_io)
    hydrogenated_io.seek(0)
    return Chem.MolFromPDBBlock(
        hydrogenated_io.read(), sanitize=False, removeHs=False)
  except ValueError as e:
    logging.warning("Unable to add hydrogens %s", e)
    raise MoleculeLoadException(e)
  finally:
    try:
      os.remove(molecule_file)
    except (OSError, TypeError):
      pass


def compute_charges(mol):
  """
  Attempt to compute Gasteiger Charges on Mol
  This also has the side effect of calculating charges on mol.
  The mol passed into this function has to already have been sanitized
  :param mol: rdkit molecule
  :return: molecule with charges
  """
  try:
    AllChem.ComputeGasteigerCharges(mol)
  except Exception as e:
    logging.exception("Unable to compute charges for mol")
    raise MoleculeLoadException(e)
  return mol


def load_molecule(molecule_file, add_hydrogens=True, calc_charges=True):
  """
  Converts molecule file to (xyz-coords, obmol object)

  Given molecule_file, returns a tuple of xyz coords of molecule
  and an rdkit object representing that molecule
  :param molecule_file: filename for molecule
  :param add_hydrogens: should add hydrogens via pdbfixer?
  :param calc_charges: should add charges vis rdkit
  :return: (xyz, mol)
  """
  if ".mol2" in molecule_file:
    my_mol = Chem.MolFromMol2File(molecule_file)
  elif ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
    my_mol = suppl[0]
  elif ".pdbqt" in molecule_file:
    pdb_block = pdbqt_to_pdb(molecule_file)
    my_mol = Chem.MolFromPDBBlock(
        str(pdb_block), sanitize=False, removeHs=False)
  elif ".pdb" in molecule_file:
    my_mol = Chem.MolFromPDBFile(
        str(molecule_file), sanitize=False, removeHs=False)
  else:
    raise ValueError("Unrecognized file type")

  if my_mol is None:
    raise ValueError("Unable to read non None Molecule Object")

  if add_hydrogens or calc_charges:
    my_mol = add_hydrogens_to_mol(my_mol)
  if calc_charges:
    compute_charges(my_mol)

  xyz = get_xyz_from_mol(my_mol)

  return xyz, my_mol


def pdbqt_file_hack_protein(mol, outfile):
  """
  Hack to convert a pdb protein into a pdbqt protein
  :param mol: rdkit Mol of protein
  :param outfile: filename which already has a valid pdb representation of mol
  """
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
  """
  Hack to convert a pdb ligand into a pdbqt ligand
  :param mol: rdkit Mol Object
  :param outfile: filename which already has a valid pdb representation of mol
  """
  PdbqtLigandWriter(mol, outfile).convert()


def write_molecule(mol, outfile, is_protein=False):
  """
   Write molecule to a file
  :param mol: rdkit Mol object
  :param outfile: filename to write mol to
  :param is_protein: is this molecule a protein?
  """
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
  elif ".sdf" in outfile:
    writer = Chem.SDWriter(outfile)
    writer.write(mol)
    writer.close()
  else:
    raise ValueError("Unsupported Format")


def pdbqt_to_pdb(filename):
  pdbqt_data = open(filename).readlines()
  pdb_block = ""
  for line in pdbqt_data:
    pdb_block += "%s\n" % line[:66]
  return pdb_block


def merge_molecules_xyz(protein_xyz, ligand_xyz):
  """Merges coordinates of protein and ligand.
  """
  return np.array(np.vstack(np.vstack((protein_xyz, ligand_xyz))))


def merge_molecules(ligand, protein):
  return Chem.rdmolops.CombineMols(ligand, protein)


class PdbqtLigandWriter(object):
  """
  Create a torsion tree and write to pdbqt file
  """

  def __init__(self, mol, outfile):
    """
    :param mol: The molecule whose value is stored in pdb format in outfile
    :param outfile: a valid pdb file with the extention .pdbqt
    """
    self.mol = mol
    self.outfile = outfile

  def convert(self):
    """
    The single public function of this class.
    It converts a molecule and a pdb file into a pdbqt file stored in outfile
    """
    import networkx as nx
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
    """
    This function does a depth first search throught he torsion tree
    :param current_partition: The current partition to expand
    :param bond: the bond which goes from the previous partition into this partition
    """
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
    """
    :param atom_number: the atom number to check for component_id
    :return: the component_id that atom_number is part of
    """
    return self.comp_map[atom_number]

  def _valid_bond(self, bond, current_partition):
    """
    used to check if a bond goes from the current partition into a partition
    that is not yet explored
    :param bond: the bond to check if it goes to an unexplored partition
    :param current_partition: the current partition of the DFS
    :return: is_valid, next_partition
    """
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
    """

    :param atom_number:
    :return:
    """
    return self.pdb_map[atom_number]

  def _create_component_map(self, components):
    """
    Creates a Map From atom_idx to disconnected_component_id
    :param components:
    :return:
    """
    comp_map = {}
    for i in range(self.mol.GetNumAtoms()):
      for j in range(len(components)):
        if i in components[j]:
          comp_map[i] = j
          break
    self.comp_map = comp_map

  def _create_pdb_map(self):
    """
    create self.pdb_map.  This is a map from rdkit atom number to
    its line in the pdb file.  We also add the two additional columns
    required for pdbqt (charge, symbol)

    note rdkit atoms are 0 indexes and pdb files are 1 indexed
    :return:
    """
    lines = [x.strip() for x in open(self.outfile).readlines()]
    lines = filter(lambda x: x.startswith("HETATM") or x.startswith("ATOM"),
                   lines)
    lines = [x[:66] for x in lines]
    pdb_map = {}
    for line in lines:
      my_values = line.split()
      atom_number = int(my_values[1])
      atom_symbol = my_values[2]
      atom_symbol = ''.join([i for i in atom_symbol if not i.isdigit()])
      line = line.replace("HETATM", "ATOM  ")
      line = "%s    +0.000 %s\n" % (line, atom_symbol.ljust(2))
      pdb_map[atom_number - 1] = line
    self.pdb_map = pdb_map

  def _mol_to_graph(self):
    """
    Convert self.mol into a graph representation
    atoms are nodes, and bonds are vertices
    store as self.graph
    """
    import networkx as nx
    G = nx.Graph()
    num_atoms = self.mol.GetNumAtoms()
    G.add_nodes_from(range(num_atoms))
    for i in range(self.mol.GetNumBonds()):
      from_idx = self.mol.GetBonds()[i].GetBeginAtomIdx()
      to_idx = self.mol.GetBonds()[i].GetEndAtomIdx()
      G.add_edge(from_idx, to_idx)
    self.graph = G

  def _get_rotatable_bonds(self):
    """
    https://github.com/rdkit/rdkit/blob/f4529c910e546af590c56eba01f96e9015c269a6/Code/GraphMol/Descriptors/Lipinski.cpp#L107
    Taken from rdkit source to find which bonds are rotatable
    store rotatable bonds in (from_atom, to_atom)
    """
    pattern = Chem.MolFromSmarts(
        "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("
        "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="
        "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#"
        "*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])"
        "[CH3])]")
    rdmolops.FastFindRings(self.mol)
    self.rotatable_bonds = self.mol.GetSubstructMatches(pattern)
