"""
Custom PDB class implementation.

The code below contains heavily modified parts of Jacob Durrant's
NNScore 2.0.1. The following notice is copied from the original NNScore
file:
# NNScore 2.01 is released under the GNU General Public License (see
# http://www.gnu.org/licenses/gpl.html).
# If you have any questions, comments, or suggestions, please don't
# hesitate to contact me, Jacob Durrant, at jdurrant [at] ucsd [dot]
# edu. If you use NNScore 2.01 in your work, please cite [REFERENCE
# HERE].
"""
import ast
import math
import textwrap
import warnings
import numpy as np
from deepchem.featurizers.nnscore_utils import AromaticRing
from deepchem.featurizers.nnscore_utils import Atom
from deepchem.featurizers.nnscore_utils import average_point
from deepchem.featurizers.nnscore_utils import Charged
from deepchem.featurizers.nnscore_utils import Point
from deepchem.featurizers.nnscore_utils import angle_between_three_points
from deepchem.featurizers.nnscore_utils import cross_product
from deepchem.featurizers.nnscore_utils import dihedral
from deepchem.featurizers.nnscore_utils import dot_product
from deepchem.featurizers.nnscore_utils import vector_subtraction
from deepchem.utils.save import log

__author__ = "Bharath Ramsundar and Jacob Durrant"
__license__ = "GNU General Public License"

def remove_redundant_rings(rings):
  """Filters out those rings which are supersets of other rings.

  Rings can be supersets of other rings, especially in molecules like
  polycyclic aromatic hydrocarbons. This function ensures that only
  "non-decomposable" rings remain in our list. Rings of length-0 are also
  removed.

  TODO(rbharath): There should be no rings of length zero if my
  understanding is correct. See if we can remove this check.

  Parameters
  ----------
  rings: list
    List of all rings in molecule.
  """
  # Remove rings of length 0
  rings = [ring for ring in rings if ring]
  # To remove duplicate entries, we convert rings from a list to set, and
  # then back to a list again. There's a snafu since each ring in rings is
  # itself a list (and lists are unhashable in python). To circumvent this
  # issue, we convert each ring into a string (after sorting). For example,
  # [2, 1] maps to '[1, 2]'. These strings are hashable. To recover the
  # original lists, we use ast.literal_eval.
  rings = [ast.literal_eval(ring_str) for ring_str in
           list(set([str(sorted(ring)) for ring in rings]))]
  # Use dictionary to maintain state about which rings are supersets.
  ring_dict = dict(zip(range(len(rings)), rings))

  for fst_index, fst_ring in enumerate(rings):
    for snd_index, snd_ring in enumerate(rings):
      if fst_index == snd_index:
        continue
      if (set(fst_ring).issubset(set(snd_ring))
          and snd_index in ring_dict):
        del ring_dict[snd_index]
  return ring_dict.values()

def print_warning(atom, residue, need, verbose=False):
  """
  Prints warning if residue has improper structure.

  Parameters
  ----------
  atom: string
    Name of affected atom.
  residue: string
    Name of affected residue.
  need: string
    Description of need for this atom in residue.
  """
  text = ('WARNING: There is no atom named "%s"' % atom +
          'in the protein residue ' + residue + '.' + ' '
          'Please use standard naming conventions for all ' +
          'protein residues. This atom is needed to determine ' +
          '%s. If this residue is far from the ' % need +
          'active site, this warning may not affect the NNScore.')
  lines = textwrap.wrap(text, 80)
  if verbose:
    for line in lines:
      print line
    print


def bond_length(element1, element2):
  """
  Returns approximate bond-length between atoms of element1 and element2.

  Bond lengths taken from Handbook of Chemistry and Physics.  The
  information provided there was very specific, so representative
  examples were used to specify the bond lengths.  Sitautions could
  arise where these lengths would be incorrect, probably slight errors
  (<0.06) in the hundreds.

  Parameters
  ----------
  element1: string:
    Name of first element.
  element2: string
    Name of second element.
  """
  # All distances are in Angstroms. Duplicate pairs not specified. For
  # example, to find distance ("H", "C"), the lookup key is ("C", "H")
  distances = {
      ("C", "C"): 1.53,
      ("N", "N"): 1.425,
      ("O", "O"): 1.469,
      ("S", "S"): 2.048,
      ("SI", "SI"): 2.359,

      ("C", "H"): 1.059,
      ("C", "N"): 1.469,
      ("C", "O"): 1.413,
      ("C", "S"): 1.819,
      ("C", "F"): 1.399,
      ("C", "CL"): 1.790,
      ("C", "BR"): 1.910,
      ("C", "I"): 2.162,

      ("N", "H"): 1.009,
      ("N", "O"): 1.463,
      ("N", "BR"): 1.843,
      ("N", "CL"): 1.743,
      ("N", "F"): 1.406,
      ("N", "I"): 2.2,

      ("O", "S"): 1.577,
      ("O", "H"): 0.967,

      # This one not from source sited above. Not sure where it's from, but
      # it wouldn't ever be used in the current context ("AutoGrow")
      ("S", "H"): 2.025/1.5,
      ("S", "N"): 1.633,
      ("S", "BR"): 2.321,
      ("S", "CL"): 2.283,
      ("S", "F"): 1.640,
      ("S", "I"): 2.687,

      ("P", "BR"): 2.366,
      ("P", "CL"): 2.008,
      ("P", "F"): 1.495,
      ("P", "I"): 2.490,
      # estimate based on eye balling Handbook of Chemistry and Physics
      ("P", "O"): 1.6,


      ("SI", "BR"): 2.284,
      ("SI", "CL"): 2.072,
      ("SI", "F"): 1.636,
      ("SI", "P"): 2.264,
      ("SI", "S"): 2.145,
      ("SI", "C"): 1.888,
      ("SI", "N"): 1.743,
      ("SI", "O"): 1.631,

      ("H", "H"): .7414,
  }
  if (element1, element2) in distances:
    return distances[(element1, element2)]
  elif (element2, element1) in distances:
    return distances[(element2, element1)]
  else:
    raise ValueError("Distance between %s and %s is unknown" %
                     (element1, element2))

class MultiStructure(object):
  """
  Handler for PDB files with multiple models.

  The output from autodock vina provides multiple output conformers in the
  generated file. This class handles this type of output.
  """

  def __init__(self):
    self.molecules = {}

  def _separate_into_models(self, lines, noisy):
    """Separate lines into a list of models."""
    if noisy:
      print "len(lines)"
      print len(lines)
      for line in lines:
        print line
    models = []
    current = None
    for line in lines:
      if "MODEL" in line and current is None:
        current = [line]
      elif "ENDMDL" in line and current is not None:
        current.append(line)
        models.append(current)
        current = None
      elif current is not None:
        current.append(line)
    if noisy:
      print "len(models)"
      print len(models)
    return models

  def load_from_files(self, pdb_filename, pdbqt_filename):
    """Loads a collection of molecular structures from input files."""
    with open(pdbqt_filename, "r") as f:
      pdbqt_lines = f.readlines()
    with open(pdb_filename, "r") as g:
      pdb_lines = g.readlines()
    pdb_models = self._separate_into_models(pdb_lines, True)
    pdbqt_models = self._separate_into_models(pdbqt_lines, False)
    assert len(pdb_models) == len(pdbqt_models)
    for index, (pdb_lines, pdbqt_lines) in enumerate(
        zip(pdb_models, pdbqt_models)):
      self.molecules[index] = PDB()
      self.molecules[index].load_from_lines(pdb_lines, pdbqt_lines)

class PDB(object):
  """
  PDB file handler class.

  TODO(rbharath): The class name here is misleading. This class actually
  depends on both the pdb and pdbqt files.

  Provides functionality for loading PDB files. Performs a number of
  clean-up and annotation steps (filling in missing bonds, identifying
  aromatic rings, charged groups, and protein secondary structure
  assignation).
  """

  def __init__(self):
    self.all_atoms = {}
    self.non_protein_atoms = {}
    self.max_x = -9999.99
    self.min_x = 9999.99
    self.max_y = -9999.99
    self.min_y = 9999.99
    self.max_z = -9998.99
    self.min_z = 9999.99
    self.rotatable_bonds_count = 0
    self.protein_resnames = [
        "ALA", "ARG", "ASN", "ASP", "ASH", "ASX",
        "CYS", "CYM", "CYX", "GLN", "GLU", "GLH", "GLX", "GLY", "HIS",
        "HID", "HIE", "HIP", "HSE", "HSD", "ILE", "LEU", "LYS", "LYN", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    self.aromatic_rings = []
    self.charges = [] # a list of objects of type charge (defined below)


  def load_from_files(self, pdb_filename, pdbqt_filename):
    """Loads this molecule from files.

    This function require both a pdbqt and pdb file (which must shared
    atomnames and indices). The reason for this dual requirement is that
    the pdbqt contains partial-charge information (which the pdb doesn't),
    while the pdb contains bond information (which the pdbqt doesn't).

    Parameters
    ----------
    pdb_filename: string
      Name of pdb file.
    pdbqt_filename: string
      Name of pdbqt file.
    """
    with open(pdbqt_filename, "r") as f:
      pdbqt_lines = f.readlines()
    with open(pdb_filename, "r") as f:
      pdb_lines = f.readlines()
    self.load_from_lines(pdb_lines, pdbqt_lines)

  def load_from_lines(self, pdb_lines, pdbqt_lines):
    """Loads the molecule from lines rather than files."""
    # Reset internal state
    self.__init__()
    # Now load the file into a list
    self.load_atoms_from_pdbqt_lines(pdbqt_lines)
    self.load_bonds_from_pdb_lines(pdb_lines)
    self.check_protein_format()
    self.assign_ligand_aromatics()
    self.assign_protein_aromatics()
    self.assign_non_protein_charges()
    self.assign_protein_charges()
    self.assign_secondary_structure()

  def load_atoms_from_pdbqt(self, pdbqt_filename):
    """Loads atoms and charges from provided PDBQT file.

    Parameters
    ----------
    pdbqt_filename: string
      Name of pdbqt file.
    """
    with open(pdbqt_filename, "r") as f:
      pdbqt_lines = f.readlines()
    self.load_atoms_from_pdbqt_lines(pdbqt_lines)

  def load_atoms_from_pdbqt_lines(self, lines):
    """Loads atoms and charges from provided PDBQT lines.

    TODO(rbharath): I'm no longer sure that this stateful paradigm is the
    right way to do things. Can I make this functional?

    Parameters
    ----------
    lines: list
      List of lines in pdbqt file
    """
    autoindex = 1
    atom_already_loaded = []

    for line in lines:
      if "between atoms" in line and " A " in line:
        self.rotatable_bonds_count = self.rotatable_bonds_count + 1

      if len(line) >= 7:
        # Load atom data (coordinates, etc.)
        if line[0:4] == "ATOM" or line[0:6] == "HETATM":
          cur_atom = Atom()
          cur_atom.read_atom_pdb_line(line)

          # this string unique identifies each atom
          key = (cur_atom.atomname.strip() + "_" +
                 str(cur_atom.resid) + "_" + cur_atom.residue.strip() +
                 "_" + cur_atom.chain.strip())

          # so each atom can only be loaded once. No rotamers.
          atom_already_loaded.append(key)
          # So you're actually reindexing everything here.
          self.all_atoms[autoindex] = cur_atom
          #### TODO(rbharath): Disabling loading of non
          if cur_atom.residue[-3:] not in self.protein_resnames:
            self.non_protein_atoms[autoindex] = cur_atom

          autoindex = autoindex + 1

  def load_bonds_from_pdb(self, pdb_filename):
    """Loads bonds from PDB file.

    Parameters
    ----------
    pdb_filename: string
      Name of pdb file.
    """
    with open(pdb_filename, "r") as f:
      lines = f.readlines()
    self.load_bonds_from_pdb_lines(lines)

  def load_bonds_from_pdb_lines(self, pdb_lines):
    """
    Loads bonds from PDB file.

    Bonds in PDBs are represented by CONECT statements. These lines follow
    the following record format:

    (see ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_Letter.pdf)

    Columns    DataType    Definition
    ---------------------------------
    1  - 6      String          -
    7  - 11     Int         Atom index.
    12 - 16     Int         Index of bonded atom.
    17 - 21     Int         Index of bonded atom.
    22 - 26     Int         Index of bonded atom.
    27 - 31     Int         Index of bonded atom.

    If more than 4 bonded atoms are present, then a second CONECT record
    must be specified.

    Parameters
    ----------
    pdb_filename: string
      Name of pdb file.
    Raises
    ------
    ValueError: On improperly formatted input.
    """
    for line in pdb_lines:
      if "CONECT" in line:
        if len(line) < 31:
          log(
              "Bad PDB! Improperly formatted CONECT line (too short)")
          continue
        atom_index = int(line[6:11].strip())
        if atom_index not in self.all_atoms:
          log(
              "Bad PDB! Improper CONECT line: (atom index not loaded)")
          continue
        bonded_atoms = []
        ranges = [(11, 16), (16, 21), (21, 26), (26, 31)]
        misformatted = False
        for (lower, upper) in ranges:
          # Check that the range is nonempty.
          if line[lower:upper].strip():
            index = int(line[lower:upper])
            if index not in self.all_atoms:
              log(
                  "Bad PDB! Improper CONECT line: (bonded atom not loaded)")
              misformatted = True
              break
            bonded_atoms.append(index)
        if misformatted:
          continue
        atom = self.all_atoms[atom_index]
        atom.add_neighbor_atom_indices(bonded_atoms)

  def save_pdb(self, filename):
    """
    Writes a PDB file version of self to filename.

    Parameters
    ----------
    filename: string
      path to desired PDB file output.
    """
    f = open(filename, 'w')
    towrite = self.save_pdb_string()
    # just so no PDB is empty, VMD will load them all
    if towrite.strip() == "":
      towrite = "ATOM      1  X   XXX             0.000   0.000   0.000                       X"
    f.write(towrite)
    f.close()

  def save_pdb_string(self):
    """
    Generates a PDB string version of self. Used by SavePDB.
    """
    to_output = ""
    # write coordinates
    for atomindex in self.all_atoms:
      to_output = (
          to_output + self.all_atoms[atomindex].create_pdb_line(atomindex)
          + "\n")
    return to_output

  def add_new_atom(self, atom):
    """
    Adds an extra atom to this PDB.

    Parameters
    ----------
    atom: object of atom class
      Will be added to self.
    """
    self.all_atoms[len(self.all_atoms.keys()) + 1] = atom

  def add_new_atoms(self, atoms):
    """
    Convenience function to add many atoms.

    Parameters
    ----------
    atoms: list
      Entries in atoms should be objects of type atom.
    """
    for atom_obj in atoms:
      self.add_new_atom(atom_obj)

  def add_new_non_protein_atom(self, atom):
    """
    Adds an extra non-protein atom to this PDB.

    Parameters
    ----------
    atom: object of atom class
      Will be added to self.
    """
    # first get available index
    ind = len(self.all_atoms.keys()) + 1
    # now add atom
    self.all_atoms[ind] = atom
    # Add to non-protein list
    self.non_protein_atoms[ind] = atom


  def connected_atoms(self, index, con_element):
    """
    Returns indices of all neighbors of atom at index of given elt.

    Parameters
    ----------
    index: integer
      Index of base atom.
    con_element: string
      Name of desired element.
    """
    atom = self.all_atoms[index]
    connected = []
    for con_index in atom.indices_of_atoms_connecting:
      con_atom = self.all_atoms[con_index]
      if con_atom.element == con_element:
        connected.append(con_index)
    return connected

  def connected_heavy_atoms(self, index):
    """
    Returns indices of all connected heavy atoms.

    Parameters
    ----------
    index: integer
      Index of base atom.
    """
    atom = self.all_atoms[index]
    connected = []
    for con_index in atom.indices_of_atoms_connecting:
      con_atom = self.all_atoms[con_index]
      if con_atom.element != "H":
        connected.append(con_index)
    return connected

  def check_protein_format(self):
    """Check that loaded protein structure is self-consistent.

    Helper function called when loading PDB from file.
    """
    for key, residue in self.get_residues().iteritems():
      residue_names = [self.all_atoms[ind].atomname.strip() for ind in residue]
      self.check_protein_format_process_residue(residue_names, key)

  def check_protein_format_process_residue(self, residue_atoms, key):
    """
    Check that specified residue in PDB is formatted correctly.

    TODO(rbharath): Lots of code repeating in this function. Factor this
    out.

    Parameters
    ----------
    residue_atoms: list
      List of atom names in residue.
    key: string
      Should be in format RESNAME_RESNUMBER_CHAIN
    """
    resname, _, _ = key.strip().split("_")
    real_resname = resname[-3:]

    if real_resname in self.protein_resnames:
      if "N" not in residue_atoms:
        print_warning("N", key, "secondary structure")
      if "C" not in residue_atoms:
        print_warning("C", key, "secondary structure")
      if "CA" not in residue_atoms:
        print_warning("CA", key, "secondary structure")

      if (real_resname == "GLU" or real_resname == "GLH"
          or real_resname == "GLX"):
        if "OE1" not in residue_atoms:
          print_warning("OE1", key, "salt-bridge interactions")
        if "OE2" not in residue_atoms:
          print_warning("OE2", key, "salt-bridge interactions")

      if (real_resname == "ASP" or real_resname == "ASH"
          or real_resname == "ASX"):
        if "OD1" not in residue_atoms:
          print_warning("OD1", key, "salt-bridge interactions")
        if "OD2" not in residue_atoms:
          print_warning("OD2", key, "salt-bridge interactions")

      if real_resname == "LYS" or real_resname == "LYN":
        if "NZ" not in residue_atoms:
          print_warning(
              "NZ", key, "pi-cation and salt-bridge interactions")

      if real_resname == "ARG":
        if "NH1" not in residue_atoms:
          print_warning(
              "NH1", key, "pi-cation and salt-bridge interactions")
        if "NH2" not in residue_atoms:
          print_warning(
              "NH2", key, "pi-cation and salt-bridge interactions")

      if (real_resname == "HIS" or real_resname == "HID"
          or real_resname == "HIE" or real_resname == "HIP"):
        if "NE2" not in residue_atoms:
          print_warning(
              "NE2", key, "pi-cation and salt-bridge interactions")
        if "ND1" not in residue_atoms:
          print_warning(
              "ND1", key, "pi-cation and salt-bridge interactions")

      if real_resname == "PHE":
        if "CG" not in residue_atoms:
          print_warning("CG", key, "pi-pi and pi-cation interactions")
        if "CD1" not in residue_atoms:
          print_warning("CD1", key, "pi-pi and pi-cation interactions")
        if "CD2" not in residue_atoms:
          print_warning("CD2", key, "pi-pi and pi-cation interactions")
        if "CE1" not in residue_atoms:
          print_warning("CE1", key, "pi-pi and pi-cation interactions")
        if "CE2" not in residue_atoms:
          print_warning("CE2", key, "pi-pi and pi-cation interactions")
        if "CZ" not in residue_atoms:
          print_warning("CZ", key, "pi-pi and pi-cation interactions")

      if real_resname == "TYR":
        if "CG" not in residue_atoms:
          print_warning("CG", key, "pi-pi and pi-cation interactions")
        if "CD1" not in residue_atoms:
          print_warning("CD1", key, "pi-pi and pi-cation interactions")
        if "CD2" not in residue_atoms:
          print_warning("CD2", key, "pi-pi and pi-cation interactions")
        if "CE1" not in residue_atoms:
          print_warning("CE1", key, "pi-pi and pi-cation interactions")
        if "CE2" not in residue_atoms:
          print_warning("CE2", key, "pi-pi and pi-cation interactions")
        if "CZ" not in residue_atoms:
          print_warning("CZ", key, "pi-pi and pi-cation interactions")

      if real_resname == "TRP":
        if "CG" not in residue_atoms:
          print_warning("CG", key, "pi-pi and pi-cation interactions")
        if "CD1" not in residue_atoms:
          print_warning("CD1", key, "pi-pi and pi-cation interactions")
        if "CD2" not in residue_atoms:
          print_warning("CD2", key, "pi-pi and pi-cation interactions")
        if "NE1" not in residue_atoms:
          print_warning("NE1", key, "pi-pi and pi-cation interactions")
        if "CE2" not in residue_atoms:
          print_warning("CE2", key, "pi-pi and pi-cation interactions")
        if "CE3" not in residue_atoms:
          print_warning("CE3", key, "pi-pi and pi-cation interactions")
        if "CZ2" not in residue_atoms:
          print_warning("CZ2", key, "pi-pi and pi-cation interactions")
        if "CZ3" not in residue_atoms:
          print_warning("CZ3", key, "pi-pi and pi-cation interactions")
        if "CH2" not in residue_atoms:
          print_warning("CH2", key, "pi-pi and pi-cation interactions")

      if (real_resname == "HIS" or real_resname == "HID" or
          real_resname == "HIE" or real_resname == "HIP"):
        if "CG" not in residue_atoms:
          print_warning("CG", key, "pi-pi and pi-cation interactions")
        if "ND1" not in residue_atoms:
          print_warning("ND1", key, "pi-pi and pi-cation interactions")
        if "CD2" not in residue_atoms:
          print_warning("CD2", key, "pi-pi and pi-cation interactions")
        if "CE1" not in residue_atoms:
          print_warning("CE2", key, "pi-pi and pi-cation interactions")
        if "NE2" not in residue_atoms:
          print_warning("NE2", key, "pi-pi and pi-cation interactions")


  # Functions to determine the bond connectivity based on distance
  # ==============================================================

  # Functions to identify positive charges
  # ======================================

  def identify_metallic_charges(self):
    """Assign charges to metallic ions.

    Returns
    -------
    charges: list
      Contains a Charge object for every metallic cation.
    """
    # Metallic atoms are assumed to be cations.
    charges = []
    for atom_index in self.non_protein_atoms:
      atom = self.non_protein_atoms[atom_index]
      if (atom.element == "MG" or atom.element == "MN" or
          atom.element == "RH" or atom.element == "ZN" or
          atom.element == "FE" or atom.element == "BI" or
          atom.element == "AS" or atom.element == "AG"):
        chrg = Charged(atom.coordinates, [atom_index], True)
        charges.append(chrg)
    return charges

  def identify_nitrogen_charges(self):
    """Assign charges to nitrogen groups where necessary.

    Returns
    -------
    charges: list
      Contains a Charge object for every charged nitrogen group.
    """
    charges = []
    for atom_index in self.non_protein_atoms:
      atom = self.non_protein_atoms[atom_index]
      # Get all the quartenary amines on non-protein residues (these are the
      # only non-protein groups that will be identified as positively
      # charged). Note that nitrogen has only 5 valence electrons (out of 8
      # for a full shell), so any nitrogen with four bonds must be positively
      # charged (think NH4+).
      if atom.element == "N":
        # a quartenary amine, so it's easy
        if atom.number_of_neighbors() == 4:
          indexes = [atom_index]
          indexes.extend(atom.indices_of_atoms_connecting)
          # so the indices stored is just the index of the nitrogen and any
          # attached atoms
          chrg = Charged(atom.coordinates, indexes, True)
          charges.append(chrg)
        # maybe you only have two hydrogens added, but they're sp3 hybridized.
        # Just count this as a quartenary amine, since I think the positive
        # charge would be stabilized. This situation can arise with
        # lone-pair electron nitrogen compounds like pyrrolidine
        # (http://www.chem.ucla.edu/harding/tutorials/lone_pair.pdf)
        elif atom.number_of_neighbors() == 3:
          nitrogen = atom
          atom1 = self.all_atoms[atom.indices_of_atoms_connecting[0]]
          atom2 = self.all_atoms[atom.indices_of_atoms_connecting[1]]
          atom3 = self.all_atoms[atom.indices_of_atoms_connecting[2]]
          angle1 = (angle_between_three_points(
              atom1.coordinates, nitrogen.coordinates, atom2.coordinates)
                    * 180.0 / math.pi)
          angle2 = (angle_between_three_points(
              atom1.coordinates, nitrogen.coordinates, atom3.coordinates)
                    * 180.0 / math.pi)
          angle3 = (angle_between_three_points(
              atom2.coordinates, nitrogen.coordinates, atom3.coordinates)
                    * 180.0 / math.pi)
          average_angle = (angle1 + angle2 + angle3) / 3
          # Test that the angles approximately match the tetrahedral 109
          # degrees
          if math.fabs(average_angle - 109.0) < 5.0:
            indexes = [atom_index]
            indexes.extend(atom.indices_of_atoms_connecting)
            # so indexes added are the nitrogen and any attached atoms.
            chrg = Charged(nitrogen.coordinates, indexes, True)
            charges.append(chrg)
    return charges

  def identify_phosphorus_charges(self):
    """Assign charges to phosphorus groups where necessary.

    Searches for phosphate-like groups and assigns charges.

    Returns
    -------
    charges: list
      Contains a Charge object for every charged phosphorus group.
    """
    charges = []
    for atom_index in self.non_protein_atoms:
      atom = self.non_protein_atoms[atom_index]
      # let's check for a phosphate or anything where a phosphorus is bound
      # to two oxygens, where both oxygens are bound to only one heavy atom
      # (the phosphorus). I think this will get several phosphorus
      # substances.
      if atom.element == "P":
        oxygens = self.connected_atoms(atom_index, "O")
        if len(oxygens) >= 2: # the phosphorus is bound to at least two oxygens
          # now count the number of oxygens bound only to the phosphorus
          count = 0
          for oxygen_index in oxygens:
            if len(self.connected_heavy_atoms(oxygen_index)) == 1:
              count = count + 1
          if count >= 2:
            indexes = [atom_index]
            indexes.extend(oxygens)
            chrg = Charged(atom.coordinates, indexes, False)
            charges.append(chrg)
    return charges

  def identify_carbon_charges(self):
    """Assign charges to carbon groups where necessary.

    Checks for guanidino-like groups and carboxylates.

    TODO(rbharath): This function is monolithic and very special-purpose.
    Can some more general design be created here?

    Returns
    -------
    charges: list
      Contains a Charge object for every charged carbon group.
    """
    charges = []
    for atom_index in self.non_protein_atoms:
      atom = self.non_protein_atoms[atom_index]
      # let's check for guanidino-like groups (actually H2N-C-NH2,
      # where not CN3.)
      if atom.element == "C":
        # if the carbon has only three atoms connected to it
        if atom.number_of_neighbors() == 3:
          nitrogens = self.connected_atoms(atom_index, "N")
          # if true, carbon is connected to at least two nitrogens now,
          # so we need to count the number of nitrogens that are only
          # connected to one heavy atom (the carbon)
          if len(nitrogens) >= 2:

            nitrogens_to_use = []
            all_connected = atom.indices_of_atoms_connecting[:]
            # Index of atom that connects this charged group to
            # the rest of the molecule, ultimately to make sure
            # it's sp3 hybridized. Remains -1 if no such atom exists.
            connector_ind = -1

            for atmindex in nitrogens:
              if len(self.connected_heavy_atoms(atmindex)) == 1:
                nitrogens_to_use.append(atmindex)
                all_connected.remove(atmindex)

            # TODO(rbharath): Is picking the first non-nitrogen atom
            # correct here?
            if len(all_connected) > 0:
              connector_ind = all_connected[0]

            # Handle case of guanidinium cation
            if len(nitrogens_to_use) == 3 and connector_ind == -1:
              charges.append(Charged(atom.coordinates.copy_of(),
                                     [atom_index], True))
            elif len(nitrogens_to_use) == 2 and connector_ind != -1:
              # so there are at two nitrogens that are only
              # connected to the carbon (and probably some
              # hydrogens)

              # now you need to make sure connector_ind atom is sp3 hybridized
              connector_atom = self.all_atoms[connector_ind]
              if ((connector_atom.element == "C" and
                   connector_atom.number_of_neighbors() == 4)
                  or (connector_atom.element == "O"
                      and connector_atom.number_of_neighbors() == 2)
                  or connector_atom.element == "N"
                  or connector_atom.element == "S"
                  or connector_atom.element == "P"):

                # There are only two "guanidino" nitrogens. Assume the
                # negative charge is spread equally between the two.
                avg_pt = average_point(
                    [self.all_atoms[nitrogen].coordinates for nitrogen in
                     nitrogens_to_use])

                indexes = [atom_index]
                indexes.extend(nitrogens_to_use)
                indexes.extend(self.connected_atoms(
                    nitrogens_to_use[0], "H"))
                indexes.extend(self.connected_atoms(
                    nitrogens_to_use[1], "H"))
                charges.append(Charged(avg_pt, indexes, True))

      if atom.element == "C": # let's check for a carboxylate
          # a carboxylate carbon will have three items connected to it.
          if atom.number_of_neighbors() == 3:
            oxygens = self.connected_atoms(atom_index, "O")
            # a carboxylate will have two oxygens connected to
            # it. Now, each of the oxygens should be connected
            # to only one heavy atom (so if it's connected to a
            # hydrogen, that's okay)
            if len(oxygens) == 2:
              if (len(self.connected_heavy_atoms(oxygens[0])) == 1
                  and len(self.connected_heavy_atoms(oxygens[1])) == 1):
                # so it's a carboxylate! Add a negative charge.

                # Assume negative charge is centered between the two
                # oxygens.
                avg_pt = average_point(
                    [self.all_atoms[oxygen].coordinates for oxygen in oxygens])
                chrg = Charged(
                    avg_pt, [oxygens[0], atom_index, oxygens[1]], False)
                charges.append(chrg)
    return charges

  def identify_sulfur_charges(self):
    """Assigns charges to sulfur groups.

    Searches for Sulfonates.

    Returns
    -------
    charges: list
      Contains a Charge object for every charged sulfur group.
    """
    charges = []
    for atom_index in self.non_protein_atoms:
      atom = self.non_protein_atoms[atom_index]
      # let's check for a sulfonate or anything where a sulfur is
      # bound to at least three oxygens and at least three are
      # bound to only the sulfur (or the sulfur and a hydrogen).
      if atom.element == "S":
        oxygens = self.connected_atoms(atom_index, "O")
        # the sulfur is bound to at least three oxygens now
        # count the number of oxygens that are only bound to the
        # sulfur
        if len(oxygens) >= 3:
          count = 0
          for oxygen_index in oxygens:
            if len(self.connected_heavy_atoms(oxygen_index)) == 1:
              count = count + 1
          # so there are at least three oxygens that are only
          # bound to the sulfur
          if count >= 3:
            indexes = [atom_index]
            indexes.extend(oxygens)
            chrg = Charged(atom.coordinates, indexes, False)
            charges.append(chrg)
    return charges

  def assign_non_protein_charges(self):
    """
    Assign positive and negative charges to non-protein atoms.

    This function handles the following cases:

      1) Metallic ions (assumed to be cations)
      2) Quartenary amines (such as NH4+)
      2) sp3 hybridized nitrogen (such as pyrrolidine)
      3) Carboxylates (RCOO-)
      4) Guanidino Groups (NHC(=NH)NH2)
      5) Phosphates (PO4(3-))
      6) Sulfonate (RSO2O-)
    """
    self.charges += self.identify_metallic_charges()
    self.charges += self.identify_nitrogen_charges()
    self.charges += self.identify_carbon_charges()
    self.charges += self.identify_phosphorus_charges()
    self.charges += self.identify_sulfur_charges()

  def get_residues(self):
    """Returns a dictionary containing all residues in this protein.

    The generated dictionary uses keys of the following type to uniquely identify
    protein residues: RESNAME_RESNUMBER_CHAIN.

    Returns
    -------
    residues: dictionary
      Each key is of type defined above and each value is a list of the
      atom-indices that make up this residue.
    """
    residues = {}
    # Group atoms in the same residue together
    for atom_index in self.all_atoms:
      atom = self.all_atoms[atom_index]
      # Assign each atom a residue key.
      key = atom.residue + "_" + str(atom.resid) + "_" + atom.chain
      if key not in residues:
        residues[key] = []
      residues[key].append(atom_index)

    # Handle edge case of last residue.
    return residues


  def assign_protein_charges(self):
    """Assigns charges to atoms in charged residues.
    """
    residues = self.get_residues()
    self.charges += self.get_lysine_charges(residues)
    self.charges += self.get_arginine_charges(residues)
    self.charges += self.get_histidine_charges(residues)
    self.charges += self.get_glutamic_acid_charges(residues)
    self.charges += self.get_aspartic_acid_charges(residues)

  def get_residue_charges(self, residues, resnames, atomnames,
                          charged_atomnames, positive=True):
    """Helper function that assigns charges to specified residue.

    Regardless of protonation state, we assume below that residues are
    charged, since evidence in the literature ("The Cation Pi Interaction,"
    TODO(rbharath): Verify citation) suggests that charges will be
    stabilized.

    Parameters
    ---------
    residues: dictionary
      Dict output by get_residue_list
    resnames: list
      List of acceptable names for residue (e.g. [PHE], [HIS, HIP, HIE,
      HID])
    atomnames: list
      List of names of atoms in charged group.
    charged_atomnames: list
      List of atoms which will be averaged to yield charge location.
    positive: bool
      Whether charge is positive or not.
    Returns
    -------
    aromatics: list
      List of Aromatic objects.
    """
    charges = []
    for key, res in residues.iteritems():
      resname, _, _ = key.strip().split("_")
      real_resname = resname[-3:]
      if real_resname in resnames:
        indices = []
        charged_atoms = [] # The terminal nitrogen holds charge.
        # Select those atoms which are part of the charged group.
        for index in res:
          atom = self.all_atoms[index]
          atomname = atom.atomname.strip()
          if atomname in atomnames:
            indices.append(index)
          if atomname in charged_atomnames:
            charged_atoms.append(atom)
        if len(charged_atoms) == len(charged_atomnames):
          avg_pt = average_point([n.coordinates for n in charged_atoms])
          if avg_pt.magnitude() != 0:
            charges.append(Charged(avg_pt, indices, positive))
    return charges

  def get_lysine_charges(self, residues):
    """Assign charges to lysine residues.

    Regardless of protonation state, assume that lysine is charged.
    Recall that LYS is positive charged lysine and LYN is neutral. See
    http://www.cgl.ucsf.edu/chimera/docs/ContributedSoftware/addh/addh.html

    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    """
    return self.get_residue_charges(
        residues, ["LYS", "LYN"],
        ["NZ", "HZ1", "HNZ1", "HZ2", "HNZ2", "HZ3", "HNZ3"],
        ["NZ"])

  def get_arginine_charges(self, residues):
    """Assign charges to arginine residues.

    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    """
    return self.get_residue_charges(
        residues, ["ARG"],
        ["NH1", "NH2", "2HH2", "HN22", "1HH2", "HN12", "CZ", "2HH1", "HN21",
         "1HH1", "HN11"], ["NH1", "NH2"])

  def get_histidine_charges(self, residues):
    """Assign charges to histidine residues.

    The specific histidine name determines the protonation state:

    * HID: Protonate delta-Nitrogen.
    * HIE: Protonate epsilon-Nitrogen.
    * HIP: Protonate both nitrogens.
    * HIS: Protonation unspecified.

    Regardless of protonation state, assume it's charged. This based on
    "The Cation-Pi Interaction," which suggests protonated state would
    be stabilized. But let's not consider HIS when doing salt bridges.
    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    """
    return self.get_residue_charges(
        residues, ["HIS", "HID", "HIE", "HIP"],
        ["NE2", "ND1", "HE2", "HD1", "CE1", "CD2", "CG"],
        ["NE2", "ND1"])

  def get_glutamic_acid_charges(self, residues):
    """Assign charges to histidine residues.

    The specific glutamic acid name determines the protonation state:

    * GLU: Negatively charged (deprotonated).
    * GLH: Neutral charge (protonated).
    * GLX: Protonation unspecified.

    See
    http://aria.pasteur.fr/documentation/use-aria/version-2.2/non-standard-atom-or-residue-definitions
    or
    http://proteopedia.org/wiki/index.php/Standard_Residues

    Regardless of protonation state, assume it's charged. This based on
    "The Cation-Pi Interaction," which suggests protonated state would
    be stabilized..

    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    """
    return self.get_residue_charges(
        residues, ["GLU", "GLH", "GLX"],
        ["OE1", "OE2", "CD"], ["OE1", "OE2"], positive=False)

  def get_aspartic_acid_charges(self, residues):
    """Assign charges to aspartic acid residues.

    The specific aspartic acid name determines the protonation.

    * ASP: Negatively charged (deprotonated).
    * ASH: Neutral charge (protonated).
    * ASX: Protonation unspecified.

    Regardless of protonation state, assume it's charged. This based on
    "The Cation-Pi Interaction," which suggests protonated state would
    be stabilized.
    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    """
    return self.get_residue_charges(
        residues, ["ASP", "ASH", "ASX"],
        ["OD1", "OD2", "CG"], ["OD1", "OD2"], positive=False)

  # Functions to identify aromatic rings
  # ====================================

  def get_aromatic_marker(self, indices_of_ring):
    """Identify aromatic markers.

    The aromatic marker is an object of class AromaticRing that specifies
    the aromatic ring's center, radius, indices of ring-atoms, and equation
    of aromatic plane (recall that an aromatic ring must be planar).

    Parameters
    ----------
    indices_of_ring: list
      Contains atom indices for all atoms in the ring.
    Raises
    ------
    ValueError:
      If len(indices_of_ring) < 3.  In this case, it is not possible to
      construct an aromatic marker (3 points are required to specify the
      aromatic plane). This happens most often when a residue is missing
      atoms (when the crystal structure failed to resolve an atom, it is
      often omitted from the PDB file).
    """
    if len(indices_of_ring) < 3:
      raise ValueError("3 points must be specified to compute aromatic plane")
    # first identify the center point
    points_list = []
    pos = np.array([0, 0, 0])

    for index in indices_of_ring:
      atom = self.all_atoms[index]
      points_list.append(atom.coordinates)
      pos += atom.coordinates.as_array().astype(pos.dtype)

    center = Point(coords=pos/len(indices_of_ring))
    radius = 0.0
    for index in indices_of_ring:
      atom = self.all_atoms[index]
      dist = center.dist_to(atom.coordinates)
      if dist > radius:
        radius = dist

    # now get the plane that defines this ring. Recall that there are
    # atleast 3-points in indices_of_ring by ValueError above.
    ring_coords = lambda i: self.all_atoms[indices_of_ring[i]].coordinates
    if len(indices_of_ring) == 3:
      indices = [0, 1, 2]
    elif len(indices_of_ring) == 4:
      indices = [0, 1, 3]
    elif len(indices_of_ring) > 4: # best, for 5 and 6 member rings
      indices = [0, 2, 4]
    a, b, c = [ring_coords(i) for i in indices]

    ab = vector_subtraction(b, a)
    ac = vector_subtraction(c, a)
    abxac = cross_product(ab, ac)

    ## formula for plane will be ax + by + cz = d
    offset = dot_product(abxac, self.all_atoms[indices_of_ring[0]].coordinates)
    return AromaticRing(center, indices_of_ring,
                        list(abxac.as_array()) + [offset], radius)

  def ring_is_flat(self, ring):
    """Checks whether specified ring is flat.

    Parameters
    ----------
    ring: list
      List of the atom indices for ring.
    """
    for ind in range(-3, len(ring)-3):
      pt1 = self.non_protein_atoms[ring[ind]].coordinates
      pt2 = self.non_protein_atoms[ring[ind+1]].coordinates
      pt3 = self.non_protein_atoms[ring[ind+2]].coordinates
      pt4 = self.non_protein_atoms[ring[ind+3]].coordinates

      # first, let's see if the last atom in this ring is a carbon
      # connected to four atoms. That would be a quick way of
      # telling this is not an aromatic ring
      cur_atom = self.non_protein_atoms[ring[ind+3]]
      if cur_atom.element == "C" and cur_atom.number_of_neighbors() == 4:
        return False

      # now check the dihedral between the ring atoms to see if
      # it's flat
      angle = dihedral(pt1, pt2, pt3, pt4) * 180 / math.pi
      # 15 degrees is the cutoff, ring[ind], ring[ind+1], ring[ind+2],
      # ring[ind+3] range of this function is -pi to pi
      if (angle > -165 and angle < -15) or (angle > 15 and angle < 165):
        return False

      # now check the dihedral between the ring atoms and an atom
      # connected to the current atom to see if that's flat too.
      for substituent_atom_index in cur_atom.indices_of_atoms_connecting:
        pt_sub = self.non_protein_atoms[substituent_atom_index].coordinates
        angle = dihedral(pt2, pt3, pt4, pt_sub) * 180 / math.pi
        # 15 degress is the cutoff, ring[ind], ring[ind+1], ring[ind+2],
        # ring[ind+3], range of this function is -pi to pi
        if (angle > -165 and angle < -15) or (angle > 15 and angle < 165):
          return False
    return True

  def assign_ligand_aromatics(self):
    """Identifies aromatic rings in ligands.
    """
    # Get all the rings containing each of the atoms in the ligand
    rings = []
    for atom_index in self.non_protein_atoms:
      rings.extend(self.all_rings_containing_atom(atom_index))

    rings = remove_redundant_rings(rings)
    # Aromatic rings are of length 5 or 6
    rings = [ring for ring in rings if len(ring) == 5 or len(ring) == 6]
    # Due to data errors in PDB files, there are cases in which
    # non-protein atoms are bonded to protein atoms. Manually remove these
    # cases, by testing that ring atom indices are a subset of non-protein
    # ring indices.
    rings = [ring for ring in rings if
             set(ring).issubset(self.non_protein_atoms.keys())]
    # Aromatic rings are flat
    rings = [ring for ring in rings if self.ring_is_flat(ring)]

    for ring in rings:
      self.aromatic_rings.append(self.get_aromatic_marker(ring))

  def all_rings_containing_atom(self, index):
    """Identify all rings that contain atom at index.

    Parameters
    ----------
    index: int
      Index of provided atom.
    """

    all_rings = []
    atom = self.all_atoms[index]
    for connected_atom in atom.indices_of_atoms_connecting:
      self.ring_recursive(connected_atom, [index], index, all_rings)
    return all_rings

  def ring_recursive(self, index, already_crossed, orig_atom, all_rings):
    """Recursive helper function for ring identification.

    Parameters
    ----------
    index: int
      Index of specified atom.
    already_crossed: list
      List of atom-indices of atoms already seen in recursive traversal of
      molecular graph.
    orig_atom: int
      Index of the original atom in ring.
    all_rings: list
      Used to recursively build up ring structure.
    """

    # Aromatic rings are of length <= 6
    if len(already_crossed) > 6:
      return

    atom = self.all_atoms[index]
    updated_crossings = already_crossed[:]
    updated_crossings.append(index)

    for connected_atom in atom.indices_of_atoms_connecting:
      if connected_atom not in already_crossed:
        self.ring_recursive(
            connected_atom, updated_crossings, orig_atom, all_rings)
      if connected_atom == orig_atom and orig_atom != already_crossed[-1]:
        all_rings.append(updated_crossings)


  def assign_protein_aromatics(self):
    """Identifies aromatic rings in protein residues.
    """
    residues = self.get_residues()
    self.aromatic_rings += self.get_phenylalanine_aromatics(residues)
    self.aromatic_rings += self.get_tyrosine_aromatics(residues)
    self.aromatic_rings += self.get_histidine_aromatics(residues)
    self.aromatic_rings += self.get_tryptophan_aromatics(residues)

  def get_residue_aromatics(self, residues, resname, ring_atomnames):
    """Helper function that identifies aromatics in given residue.

    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    resname: list
      List of acceptable names for residue (e.g. [PHE], [HIS, HIP, HIE,
      HID])
    ring_atomnames: list
      List of names of atoms in aromatic ring.
    Returns
    -------
    aromatics: list
      List of Aromatic objects.
    """
    aromatics = []
    for key, res in residues.iteritems():
      real_resname, _, _ = key.strip().split("_")
      indices_of_ring = []
      if real_resname in resname:
        indices_of_ring = []
        for index in res:
          if self.all_atoms[index].atomname.strip() in ring_atomnames:
            indices_of_ring.append(index)
        # At least 3 indices are required to identify the aromatic plane.
        if len(indices_of_ring) < 3:
          continue
        else:
          aromatics.append(self.get_aromatic_marker(indices_of_ring))
        #if self.get_aromatic_marker(indices_of_ring) is None:
        #  raise ValueError("None at %s for %s" % (key,
        #  str(indices_of_ring)))
    return aromatics


  def get_phenylalanine_aromatics(self, residues):
    """Assign aromatics in phenylalanines.

    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    Returns
    -------
    aromatics: list
      List of Aromatic objects for aromatics in phenylalanines.
    """
    return self.get_residue_aromatics(
        residues, "PHE",
        ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"])

  def get_tyrosine_aromatics(self, residues):
    """Assign aromatics in tyrosines.

    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    Returns
    -------
    aromatics: list
      List of Aromatic objects for aromatics in tyrosines.
    """
    return self.get_residue_aromatics(
        residues, "TYR", ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"])

  def get_histidine_aromatics(self, residues):
    """Assign aromatics in histidines.

    Parameters
    ----------
    residues: dictionary
      Dict output by get_residue_list
    Returns
    -------
    aromatics: list
      List of Aromatic objects for aromatics in histidines.
    """
    return self.get_residue_aromatics(
        residues, ["HIS", "HID", "HIE", "HIP"],
        ["CG", "ND1", "CE1", "NE2", "CD2"])

  def get_tryptophan_aromatics(self, residues):
    """Assign aromatics in tryptophans.

    Parameters
    ----------
    residues: list
      List of tuples output by get_residue_list
    Returns
    -------
    aromatics: list
      List of Aromatic objects for aromatics in tryptophans.
    """
    # Tryptophan has two aromatic rings.
    small_ring = self.get_residue_aromatics(
        residues, ["TRP"],
        ["CG", "CD1", "NE1", "CE2", "CD2"])
    large_ring = self.get_residue_aromatics(
        residues, ["TRP"],
        ["CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2"])
    return small_ring + large_ring

  # Functions to assign secondary structure to protein residues
  # ===========================================================

  def get_structure_dict(self):
    """Creates a dictionary of preliminary structure labels.

    Uses a simple heuristic of checking dihedral angles to classify as
    alpha helix or beta sheet.

    TODO(rbharath): This prediction function is overly simplistic and
    fails to provide reasonable results. Swap to use JPred results instead.

    Returns:
    structure: dict
      Maps keys of format RESNUMBER_CHAIN to one of ALPHA, BETA, or OTHER.
    """
    # first, we need to know what residues are available
    resids = []
    #print self.get_residues()
    for key in self.get_residues():
      _, resnum, chain = key.split("_")
      resids.append(resnum + "_" + chain)

    structure = {}
    for resid in resids:
      structure[resid] = "OTHER"

    atoms = []
    for atom_index in self.all_atoms:
      atom = self.all_atoms[atom_index]
      if atom.side_chain_or_backbone() == "BACKBONE":
        # TODO(rbharath): Why magic number 8?
        if len(atoms) < 8:
          atoms.append(atom)
        else:
          atoms.pop(0)
          atoms.append(atom)

          # now make sure the first four all have the same resid and
          # the last four all have the same resid
          # TODO(rbharath): Ugly code right here...
          if (atoms[0].resid == atoms[1].resid
              and atoms[0].resid == atoms[2].resid
              and atoms[0].resid == atoms[3].resid
              and atoms[0] != atoms[4].resid
              and atoms[4].resid == atoms[5].resid
              and atoms[4].resid == atoms[6].resid
              and atoms[4].resid == atoms[7].resid
              and atoms[0].resid + 1 == atoms[7].resid
              and atoms[0].chain == atoms[7].chain):

            resid1 = atoms[0].resid
            resid2 = atoms[7].resid

            # Now give easier to use names to the atoms
            for atom in atoms:
              atomname = atom.atomname.strip()
              if atom.resid == resid1 and atomname == "N":
                first_n = atom
              if atom.resid == resid1 and atomname == "C":
                first_c = atom
              if atom.resid == resid1 and atomname == "CA":
                first_ca = atom

              if atom.resid == resid2 and atomname == "N":
                second_n = atom
              if atom.resid == resid2 and atomname == "C":
                second_c = atom
              if atom.resid == resid2 and atomname == "CA":
                second_ca = atom

            # Now compute the phi and psi dihedral angles
            phi = (dihedral(first_c.coordinates, second_n.coordinates,
                            second_ca.coordinates, second_c.coordinates)
                   * 180.0 / math.pi)
            psi = (dihedral(first_n.coordinates, first_ca.coordinates,
                            first_c.coordinates, second_n.coordinates)
                   * 180.0 / math.pi)

            # Now use those angles to determine if it's alpha or beta
            if phi > -145 and phi < -35 and psi > -70 and psi < 50:
              key1 = str(first_c.resid) + "_" + first_c.chain
              key2 = str(second_c.resid) + "_" + second_c.chain
              structure[key1] = "ALPHA"
              structure[key2] = "ALPHA"
            if ((phi >= -180 and phi < -40 and psi <= 180 and psi > 90)
                or (phi >= -180 and phi < -70 and psi <= -165)):
              key1 = str(first_c.resid) + "_" + first_c.chain
              key2 = str(second_c.resid) + "_" + second_c.chain
              structure[key1] = "BETA"
              structure[key2] = "BETA"
    return structure

  def process_alpha_helices(self, ca_list):
    """Postprocess alpha helices to remove extraneous labels.

    TODO(rbharath): The comparison method here is quadratic. Can we do
    better with a nice datastructure?

    Parameters
    ----------
    ca_list: list
      List of all alpha carbons in protein.
    """
    change = True
    while change:
      change = False

      # A residue of index i is only going to be in an alpha helix
      # its CA is within 6 A of the CA of the residue i + 3
      for ca_atom_index in ca_list:
        ca_atom = self.all_atoms[ca_atom_index]
        if ca_atom.structure == "ALPHA":
          # so it's in an alpha helix
          another_alpha_is_close = False
          for other_ca_atom_index in ca_list:
            # so now compare that CA to all the other CA's
            other_ca_atom = self.all_atoms[other_ca_atom_index]
            # so it's also in an alpha helix
            if other_ca_atom.structure == "ALPHA":
              if (other_ca_atom.resid - 3 == ca_atom.resid
                  or other_ca_atom.resid + 3 == ca_atom.resid):
                # so this CA atom is one of the ones the first atom
                # might hydrogen bond with
                if other_ca_atom.coordinates.dist_to(ca_atom.coordinates) < 6.0:
                  # so these two CA atoms are close enough together
                  # that their residues are probably hydrogen bonded
                  another_alpha_is_close = True
                  break
          if not another_alpha_is_close:
            self.set_structure_of_residue(ca_atom.chain, ca_atom.resid, "OTHER")
            change = True

      # Alpha helices are only alpha helices if they span at least 4
      # residues (to wrap around and hydrogen bond). I'm going to
      # require them to span at least 5 residues, based on
      # examination of many structures.
      for index_in_list in range(len(ca_list)-5):

        index_in_pdb1 = ca_list[index_in_list]
        index_in_pdb2 = ca_list[index_in_list+1]
        index_in_pdb3 = ca_list[index_in_list+2]
        index_in_pdb4 = ca_list[index_in_list+3]
        index_in_pdb5 = ca_list[index_in_list+4]
        index_in_pdb6 = ca_list[index_in_list+5]

        atom1 = self.all_atoms[index_in_pdb1]
        atom2 = self.all_atoms[index_in_pdb2]
        atom3 = self.all_atoms[index_in_pdb3]
        atom4 = self.all_atoms[index_in_pdb4]
        atom5 = self.all_atoms[index_in_pdb5]
        atom6 = self.all_atoms[index_in_pdb6]

        if (atom1.resid + 1 == atom2.resid
            and atom2.resid + 1 == atom3.resid
            and atom3.resid + 1 == atom4.resid
            and atom4.resid + 1 == atom5.resid
            and atom5.resid + 1 == atom6.resid): # so they are sequential
          if (atom1.structure != "ALPHA"
              and atom2.structure == "ALPHA"
              and atom3.structure != "ALPHA"):
            self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
            change = True
          if (atom2.structure != "ALPHA"
              and atom3.structure == "ALPHA"
              and atom4.structure != "ALPHA"):
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            change = True
          if (atom3.structure != "ALPHA"
              and atom4.structure == "ALPHA"
              and atom5.structure != "ALPHA"):
            self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
            change = True
          if (atom4.structure != "ALPHA"
              and atom5.structure == "ALPHA"
              and atom6.structure != "ALPHA"):
            self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
            change = True

          if (atom1.structure != "ALPHA"
              and atom2.structure == "ALPHA"
              and atom3.structure == "ALPHA"
              and atom4.structure != "ALPHA"):
            self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            change = True
          if (atom2.structure != "ALPHA"
              and atom3.structure == "ALPHA"
              and atom4.structure == "ALPHA"
              and atom5.structure != "ALPHA"):
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
            change = True
          if (atom3.structure != "ALPHA"
              and atom4.structure == "ALPHA"
              and atom5.structure == "ALPHA"
              and atom6.structure != "ALPHA"):
            self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
            self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
            change = True

          if (atom1.structure != "ALPHA"
              and atom2.structure == "ALPHA"
              and atom3.structure == "ALPHA"
              and atom4.structure == "ALPHA"
              and atom5.structure != "ALPHA"):
            self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
            change = True
          if (atom2.structure != "ALPHA"
              and atom3.structure == "ALPHA"
              and atom4.structure == "ALPHA"
              and atom5.structure == "ALPHA"
              and atom6.structure != "ALPHA"):
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
            self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
            change = True

          if (atom1.structure != "ALPHA"
              and atom2.structure == "ALPHA"
              and atom3.structure == "ALPHA"
              and atom4.structure == "ALPHA"
              and atom5.structure == "ALPHA"
              and atom6.structure != "ALPHA"):
            self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            self.set_structure_of_residue(atom4.chain, atom4.resid, "OTHER")
            self.set_structure_of_residue(atom5.chain, atom5.resid, "OTHER")
            change = True

  def process_beta_sheets(self, ca_list):
    """Postprocess beta sheets to remove extraneous labels.

    TODO(rbharath): The comparison method here is quadratic. Can we do
    better with a nice datastructure?

    Parameters
    ----------
    ca_list: list
      List of all alpha carbons in protein.
    """
    change = True
    while change:
      change = False

      # now go through each of the BETA CA atoms. A residue is only
      # going to be called a beta sheet if CA atom is within 6.0 A
      # of another CA beta, same chain, but index difference > 2.
      for ca_atom_index in ca_list:
        ca_atom = self.all_atoms[ca_atom_index]
        if ca_atom.structure == "BETA":
          # so it's in a beta sheet
          another_beta_is_close = False
          for other_ca_atom_index in ca_list:
            if other_ca_atom_index != ca_atom_index:
              # so not comparing an atom to itself
              other_ca_atom = self.all_atoms[other_ca_atom_index]
              if other_ca_atom.structure == "BETA":
                # so you're comparing it only to other BETA-sheet atoms
                if other_ca_atom.chain == ca_atom.chain:
                  # so require them to be on the same chain. needed to
                  # indices can be fairly compared
                  if math.fabs(other_ca_atom.resid - ca_atom.resid) > 2:
                    # so the two residues are not simply adjacent to each
                    # other on the chain
                    if (ca_atom.coordinates.dist_to(
                        other_ca_atom.coordinates) < 6.0):
                      # so these to atoms are close to each other
                      another_beta_is_close = True
                      break
          if not another_beta_is_close:
            self.set_structure_of_residue(ca_atom.chain, ca_atom.resid, "OTHER")
            change = True

      # Now some more post-processing needs to be done. Do this
      # again to clear up mess that may have just been created
      # (single residue beta strand, for example)
      # Beta sheets are usually at least 3 residues long

      for index_in_list in range(len(ca_list)-3):

        index_in_pdb1 = ca_list[index_in_list]
        index_in_pdb2 = ca_list[index_in_list+1]
        index_in_pdb3 = ca_list[index_in_list+2]
        index_in_pdb4 = ca_list[index_in_list+3]

        atom1 = self.all_atoms[index_in_pdb1]
        atom2 = self.all_atoms[index_in_pdb2]
        atom3 = self.all_atoms[index_in_pdb3]
        atom4 = self.all_atoms[index_in_pdb4]

        if (atom1.resid + 1 == atom2.resid and atom2.resid + 1 ==
            atom3.resid and atom3.resid + 1 == atom4.resid):
          # so they are sequential

          if (atom1.structure != "BETA"
              and atom2.structure == "BETA"
              and atom3.structure != "BETA"):
            self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
            change = True
          if (atom2.structure != "BETA"
              and atom3.structure == "BETA"
              and atom4.structure != "BETA"):
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            change = True
          if (atom1.structure != "BETA"
              and atom2.structure == "BETA"
              and atom3.structure == "BETA"
              and atom4.structure != "BETA"):
            self.set_structure_of_residue(atom2.chain, atom2.resid, "OTHER")
            self.set_structure_of_residue(atom3.chain, atom3.resid, "OTHER")
            change = True

  def assign_secondary_structure(self):
    """Assign secondary structure labels (assuming self is a protein).

    Keys in this function have form RESNUMBER_CHAIN where CHAIN is
    the chain identifier for this molecule.
    """
    structure = self.get_structure_dict()

    # Now update each of the atoms with this structural information
    for atom_index in self.all_atoms:
      atom = self.all_atoms[atom_index]
      key = str(atom.resid) + "_" + atom.chain
      atom.structure = structure[key]

    ca_list = [] # first build a list of the indices of all the alpha carbons
    for atom_index in self.all_atoms:
      atom = self.all_atoms[atom_index]
      if (atom.residue.strip() in self.protein_resnames
          and atom.atomname.strip() == "CA"):
        ca_list.append(atom_index)

    # Use this list to perform sanity checks on alpha-helix and beta-sheet
    # labels.
    self.process_alpha_helices(ca_list)
    self.process_beta_sheets(ca_list)


  def set_structure_of_residue(self, chain, resid, structure):
    """Set structure of all atoms with specified chain and residue."""
    for atom_index in self.all_atoms:
      atom = self.all_atoms[atom_index]
      if atom.chain == chain and atom.resid == resid:
        atom.structure = structure
