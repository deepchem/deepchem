"""Utilities for handling PDBQT files."""

from typing import Dict, List, Optional, Set, Tuple
from deepchem.utils.typing import RDKitMol  # type: ignore


def pdbqt_to_pdb(filename: Optional[str] = None,
                 pdbqt_data: Optional[List[str]] = None) -> str:
    """Extracts the PDB part of a pdbqt file as a string.

    Either `filename` or `pdbqt_data` must be provided. This function
    strips PDBQT charge information from the provided input.

    Parameters
    ----------
    filename: str, optional  (default None)
        Filename of PDBQT file
    pdbqt_data: List[str], optional (default None)
        Raw list of lines containing data from PDBQT file.

    Returns
    -------
    pdb_block: str
        String containing the PDB portion of pdbqt file.
    """
    if filename is not None and pdbqt_data is not None:
        raise ValueError("Only one of filename or pdbqt_data can be provided")
    elif filename is None and pdbqt_data is None:
        raise ValueError("Either filename or pdbqt_data must be provided")
    elif filename is not None:
        pdbqt_data = open(filename).readlines()

    pdb_block = ""
    # FIXME: Item "None" of "Optional[List[str]]" has no attribute "__iter__" (not iterable)
    for line in pdbqt_data:  # type: ignore
        pdb_block += "%s\n" % line[:66]
    return pdb_block


def convert_protein_to_pdbqt(mol: RDKitMol, outfile: str) -> None:
    """Convert a protein PDB file into a pdbqt file.

    Writes the extra PDBQT terms directly to `outfile`.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
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
        atom_index = int(line[6:11])
        atom = mol.GetAtoms()[atom_index - 1]
        line = "%s    +0.000 %s\n" % (line, atom.GetSymbol().ljust(2))
        out_lines.append(line)
    with open(outfile, 'w') as fout:
        for line in out_lines:
            fout.write(line)


def _mol_to_graph(mol: RDKitMol):
    """Convert RDKit Mol to NetworkX graph

    Convert mol into a graph representation atoms are nodes, and bonds
    are vertices stored as graph

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        The molecule to convert into a graph.

    Returns
    -------
    graph: networkx.Graph
        Contains atoms indices as nodes, edges as bonds.

    Notes
    -----
    This function requires NetworkX to be installed.
    """
    try:
        import networkx as nx  # type: ignore
    except ModuleNotFoundError:
        raise ImportError("This function requires NetworkX to be installed.")

    G = nx.Graph()
    num_atoms = mol.GetNumAtoms()
    G.add_nodes_from(range(num_atoms))
    for i in range(mol.GetNumBonds()):
        from_idx = mol.GetBonds()[i].GetBeginAtomIdx()
        to_idx = mol.GetBonds()[i].GetEndAtomIdx()
        G.add_edge(from_idx, to_idx)
    return G


def _get_rotatable_bonds(mol: RDKitMol) -> List[Tuple[int, int]]:
    """
    https://github.com/rdkit/rdkit/blob/f4529c910e546af590c56eba01f96e9015c269a6/Code/GraphMol/Descriptors/Lipinski.cpp#L107

    Taken from rdkit source to find which bonds are rotatable store
    rotatable bonds in (from_atom, to_atom)

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        Ligand molecule

    Returns
    -------
    rotatable_bonds: List[List[int, int]]
        List of rotatable bonds in molecule

    Notes
    -----
    This function requires RDKit to be installed.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdmolops  # type: ignore
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    pattern = Chem.MolFromSmarts(
        "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("
        "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="
        "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#"
        "*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])"
        "[CH3])]")
    rdmolops.FastFindRings(mol)
    rotatable_bonds = mol.GetSubstructMatches(pattern)
    return rotatable_bonds


def convert_mol_to_pdbqt(mol: RDKitMol, outfile: str) -> None:
    """Writes the provided ligand molecule to specified file in pdbqt format.

    Creates a torsion tree and write to pdbqt file. The torsion tree
    represents rotatable bonds in the molecule.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        The molecule whose value is stored in pdb format in outfile
    outfile: str
        Filename for a valid pdb file with the extention .pdbqt

    Notes
    -----
    This function requires NetworkX to be installed.
    """
    try:
        import networkx as nx
    except ModuleNotFoundError:
        raise ImportError("This function requires NetworkX to be installed.")

    # Walk through the original file and extract ATOM/HETATM lines and
    # add PDBQT charge annotations.
    pdb_map = _create_pdb_map(outfile)
    graph = _mol_to_graph(mol)
    rotatable_bonds = _get_rotatable_bonds(mol)

    # Remove rotatable bonds from this molecule
    for bond in rotatable_bonds:
        graph.remove_edge(bond[0], bond[1])
    # Get the connected components now that the rotatable bonds have
    # been removed.
    components = [x for x in nx.connected_components(graph)]
    comp_map = _create_component_map(mol, components)

    used_partitions = set()
    lines = []
    # The root is the largest connected component.
    root = max(enumerate(components), key=lambda x: len(x[1]))[0]
    # Write the root component
    lines.append("ROOT\n")
    for atom in components[root]:
        lines.append(pdb_map[atom])
    lines.append("ENDROOT\n")
    # We've looked at the root, so take note of that
    used_partitions.add(root)
    for bond in rotatable_bonds:
        valid, next_partition = _valid_bond(used_partitions, bond, root,
                                            comp_map)
        if not valid:
            continue
        _dfs(used_partitions, next_partition, bond, components, rotatable_bonds,
             lines, pdb_map, comp_map)
    lines.append("TORSDOF %s" % len(rotatable_bonds))
    with open(outfile, 'w') as fout:
        for line in lines:
            fout.write(line)


def _create_pdb_map(outfile: str) -> Dict[int, str]:
    """Create a mapping from atom numbers to lines to write to pdbqt

    This is a map from rdkit atom number to its line in the pdb
    file. We also add the two additional columns required for
    pdbqt (charge, symbol).

    Note rdkit atoms are 0 indexed and pdb files are 1 indexed

    Parameters
    ----------
    outfile: str
        filename which already has a valid pdb representation of mol

    Returns
    -------
    pdb_map: Dict[int, str]
        Maps rdkit atom numbers to lines to be written to PDBQT file.
    """
    lines = [x.strip() for x in open(outfile).readlines()]
    lines = list(
        filter(lambda x: x.startswith("HETATM") or x.startswith("ATOM"), lines))
    lines = [x[:66] for x in lines]
    pdb_map = {}
    for line in lines:
        my_values = line.split()
        atom_number = int(my_values[1])
        atom_symbol = my_values[2].capitalize()
        atom_symbol = ''.join([i for i in atom_symbol if not i.isdigit()])
        line = line.replace("HETATM", "ATOM  ")
        line = "%s    +0.000 %s\n" % (line, atom_symbol.ljust(2))
        pdb_map[atom_number - 1] = line
    return pdb_map


def _create_component_map(mol: RDKitMol,
                          components: List[List[int]]) -> Dict[int, int]:
    """Creates a map from atom ids to disconnected component id

    For each atom in `mol`, maps it to the id of the component in the
    molecule. The intent is that this is used on a molecule whose
    rotatable bonds have been removed. `components` is a list of the
    connected components after this surgery.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        The molecule to find disconnected components in
    components: List[List[int]]
        List of connected components

    Returns
    -------
    comp_map: Dict[int, int]
        Maps atom ids to component ides
    """
    comp_map = {}
    for i in range(mol.GetNumAtoms()):
        for j in range(len(components)):
            if i in components[j]:
                comp_map[i] = j
                break
    return comp_map


def _dfs(used_partitions: Set[int], current_partition: int,
         bond: Tuple[int, int], components: List[List[int]],
         rotatable_bonds: List[Tuple[int, int]], lines: List[str],
         pdb_map: Dict[int, str], comp_map: Dict[int, int]) -> List[str]:
    """
    This function does a depth first search through the torsion tree

    Parameters
    ----------
    used_partions: Set[int]
        Partitions which have already been used
    current_partition: int
        The current partition to expand
    bond: Tuple[int, int]
        the bond which goes from the previous partition into this partition
    components: List[List[int]]
        List of connected components
    rotatable_bonds: List[Tuple[int, int]]
        List of rotatable bonds. This tuple is (from_atom, to_atom).
    lines: List[str]
        List of lines to write
    pdb_map: Dict[int, str]
        Maps atom numbers to PDBQT lines to write
    comp_map: Dict[int, int]
        Maps atom numbers to component numbers

    Returns
    -------
    lines: List[str]
        List of lines to write. This has more appended lines.
    """
    if comp_map[bond[1]] != current_partition:
        bond = (bond[1], bond[0])
    used_partitions.add(comp_map[bond[0]])
    used_partitions.add(comp_map[bond[1]])
    lines.append("BRANCH %4s %4s\n" % (bond[0] + 1, bond[1] + 1))
    for atom in components[current_partition]:
        lines.append(pdb_map[atom])
    for b in rotatable_bonds:
        valid, next_partition = \
          _valid_bond(used_partitions, b, current_partition, comp_map)
        if not valid:
            continue
        lines = _dfs(used_partitions, next_partition, b, components,
                     rotatable_bonds, lines, pdb_map, comp_map)
    lines.append("ENDBRANCH %4s %4s\n" % (bond[0] + 1, bond[1] + 1))
    return lines


def _valid_bond(used_partitions: Set[int], bond: Tuple[int, int],
                current_partition: int,
                comp_map: Dict[int, int]) -> Tuple[bool, int]:
    """Helper method to find next partition to explore.

    Used to check if a bond goes from the current partition into a
    partition that is not yet explored

    Parameters
    ----------
    used_partions: Set[int]
        Partitions which have already been used
    bond: Tuple[int, int]
        The bond to check if it goes to an unexplored partition.
        This tuple is (from_atom, to_atom).
    current_partition: int
        The current partition of the DFS
    comp_map: Dict[int, int]
        Maps atom ids to component ids

    Returns
    -------
    is_valid: bool
        Whether to exist the next partition or not
    next_partition: int
        The next partition to explore
    """
    part1 = comp_map[bond[0]]
    part2 = comp_map[bond[1]]
    if part1 != current_partition and part2 != current_partition:
        return False, 0
    if part1 == current_partition:
        next_partition = part2
    else:
        next_partition = part1
    return next_partition not in used_partitions, next_partition
