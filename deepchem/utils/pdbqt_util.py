"""Utilities for handling PDBQT files."""

def convert_mol_to_pdbqt(mol, outfile):
  """Convert a rdkit molecule into a pdbqt ligand

  Parameters
  ----------
  mol: rdkit Mol
    Ligand molecule
  outfile: str
    filename which already has a valid pdb representation of mol
  """
  PdbqtLigandWriter(mol, outfile).convert()


def convert_protein_to_pdbqt(mol, outfile):
  """Convert a protein PDB file into a pdbqt file.

  Parameters
  ----------
  mol: rdkit Mol
    Protein molecule
  outfile: str
    filename which already has a valid pdb representation of mol
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


class PdbqtLigandWriter(object):
  """
  Create a torsion tree and write to pdbqt file
  """

  def __init__(self, mol, outfile):
    """
    Parameters
    ----------
    mol: rdkit Mol
      The molecule whose value is stored in pdb format in outfile
    outfile: str
      Filename for a valid pdb file with the extention .pdbqt
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

    Parameters
    ----------
    current_partition: object
      The current partition to expand
    bond: object
      the bond which goes from the previous partition into this partition
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
    Parameters
    ----------
    atom_number: int
      The atom number to check for component_id

    Returns
    -------
    The component_id that atom_number is part of
    """
    return self.comp_map[atom_number]

  def _valid_bond(self, bond, current_partition):
    """
    used to check if a bond goes from the current partition into a partition
    that is not yet explored

    Parameters
    ----------
    bond: object
      the bond to check if it goes to an unexplored partition
    current_partition: object
      the current partition of the DFS

    Returns
    -------
    is_valid, next_partition
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
    Parameters
    ----------
    atom_number: int
      The atom number for this atom

    Returns
    -------
    The self.pdb_map lookup for this atom.
    """
    return self.pdb_map[atom_number]

  def _create_component_map(self, components):
    """Creates a Map From atom_idx to disconnected_component_id

    Sets self.comp_map to the computed compnent map.

    Parameters
    ----------
    components: list
      List of connected components
    """
    comp_map = {}
    for i in range(self.mol.GetNumAtoms()):
      for j in range(len(components)):
        if i in components[j]:
          comp_map[i] = j
          break
    self.comp_map = comp_map

  def _create_pdb_map(self):
    """Create self.pdb_map.

    This is a map from rdkit atom number to its line in the pdb
    file. We also add the two additional columns required for
    pdbqt (charge, symbol)

    note rdkit atoms are 0 indexes and pdb files are 1 indexed
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
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    pattern = Chem.MolFromSmarts(
        "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("
        "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="
        "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#"
        "*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])"
        "[CH3])]")
    rdmolops.FastFindRings(self.mol)
    self.rotatable_bonds = self.mol.GetSubstructMatches(pattern)
