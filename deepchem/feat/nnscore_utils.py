"""
Helper Classes and Functions for docking fingerprint computation.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar and Jacob Durrant"
__license__ = "GNU General Public License"

import math
import os
import subprocess
import numpy as np
import deepchem.utils.rdkit_util as rdkit_util


def force_partial_charge_computation(mol):
  """Force computation of partial charges for molecule.

  Parameters
  ----------
  mol: Rdkit Mol
    Molecule on which we compute partial charges.
  """
  rdkit_util.compute_charges(mol)


def pdbqt_to_pdb(input_file, output_directory):
  """Convert pdbqt file to pdb file.

  Parameters
  ----------
  input_file: String
    Path to input file.
  output_directory: String
    Path to desired output directory.
  """
  print(input_file, output_directory)
  raise ValueError("Not yet implemented")


def hydrogenate_and_compute_partial_charges(input_file,
                                            input_format,
                                            hyd_output=None,
                                            pdbqt_output=None,
                                            protein=True,
                                            verbose=True):
  """Outputs a hydrogenated pdb and a pdbqt with partial charges.

  Takes an input file in specified format. Generates two outputs:

  -) A pdb file that contains a hydrogenated (at pH 7.4) version of
     original compound.
  -) A pdbqt file that has computed Gasteiger partial charges. This pdbqt
     file is build from the hydrogenated pdb.

  TODO(rbharath): Can do a bit of refactoring between this function and
  pdbqt_to_pdb.

  Parameters
  ----------
  input_file: String
    Path to input file.
  input_format: String
    Name of input format.
  """
  mol = rdkit_util.load_molecule(
      input_file, add_hydrogens=True, calc_charges=True)[1]
  if verbose:
    print("Create pdb with hydrogens added")
  rdkit_util.write_molecule(mol, str(hyd_output), is_protein=protein)
  if verbose:
    print("Create a pdbqt file from the hydrogenated pdb above.")
  rdkit_util.write_molecule(mol, str(pdbqt_output), is_protein=protein)

  if protein:
    print("Removing ROOT/ENDROOT/TORSDOF")
    with open(pdbqt_output) as f:
      pdbqt_lines = f.readlines()
    filtered_lines = []
    for line in pdbqt_lines:

      filtered_lines.append(line)
    with open(pdbqt_output, "w") as f:
      f.writelines(filtered_lines)


class AromaticRing(object):
  """Holds information about an aromatic ring."""

  def __init__(self, center, indices, plane_coeff, radius):
    """
    Initializes an aromatic.

    Parameters
    ----------
    center: float
      Center of the ring.
    indices: list
      List of the atom indices for ring atoms.
    plane_coeff: list
      A list of elements [a, b, c, d] that define a plane by equation
      a x + b y + c z = d.
    radius: float
      Ring radius from center.
    """
    self.center = center
    self.indices = indices
    # a*x + b*y + c*z = dI think that
    self.plane_coeff = plane_coeff
    self.radius = radius


def average_point(points):
  """Returns the point with averaged coordinates of arguments.

  Parameters
  ----------
  points: list
    List of point objects.
  Returns
  -------
  pavg: Point object
    Has coordinates the arithmetic average of those of p1 and p2.
  """
  coords = np.array([0, 0, 0])
  for point in points:
    coords += point.as_array().astype(coords.dtype)
  if len(points) > 0:
    return Point(coords=coords / len(points))
  else:
    return Point(coords=coords)


class Point(object):
  """
  Simple implementation for a point in 3-space.
  """

  def __init__(self, x=None, y=None, z=None, coords=None):
    """
    Inputs can be specified either by explicitly providing x, y, z coords
    or by providing a numpy array of length 3.

    Parameters
    ----------
    x: float
      X-coord.
    y: float
      Y-coord.
    z: float
      Z-coord.
    coords: np.ndarray
      Should be of length 3 in format np.array([x, y, z])
    Raises
    ------
    ValueError: If no arguments are provided.
    """
    if x and y and z:
      #self.x, self.y, self.z = x, y, z
      self.coords = np.array([x, y, z])
    elif coords is not None:  # Implicit eval doesn't work on numpy arrays.
      #self.x, self.y, self.z = coords[0], coords[1], coords[2]
      self.coords = coords
    else:
      raise ValueError("Must specify coordinates for Point!")

  # TODO(bramsundar): Should this be __copy__?
  def copy_of(self):
    """Return a copy of this point."""
    return Point(coords=np.copy(self.coords))

  def dist_to(self, point):
    """Distance (in 2-norm) from this point to another."""
    return np.linalg.norm(self.coords - point.coords)

  def magnitude(self):
    """Magnitude of this point (in 2-norm)."""
    return np.linalg.norm(self.coords)
    #return self.dist_to(Point(coords=np.array([0, 0, 0])))

  def as_array(self):
    """Return the coordinates of this point as array."""
    #return np.array([self.x, self.y, self.z])
    return self.coords


class Atom(object):
  """
  Implements a container class for atoms. This class contains useful
  annotations about the atom.
  """

  def __init__(self,
               atomname="",
               residue="",
               coordinates=Point(coords=np.array([99999, 99999, 99999])),
               element="",
               pdb_index="",
               line="",
               atomtype="",
               indices_of_atoms_connecting=None,
               charge=0,
               resid=0,
               chain="",
               structure="",
               comment=""):
    """
    Initializes an atom.

    Assumes that atom is loaded from a PDB file.

    Parameters
    ----------
    atomname: string
      Name of atom. Note that atomname is not the same as residue since
      atomnames often have extra annotations (e.g., CG, NZ, etc).
    residue: string:
      Name of protein residue this atom belongs to.
    element: string
      Name of atom's element.
    coordinate: point
      A point object (x, y, z are in Angstroms).
    pdb_index: string
      Index of the atom in source PDB file.
    line: string
      The line in the PDB file which specifies this atom.
    atomtype: string
      Element of atom. This differs from atomname which typically has extra
      annotations (e.g. CA, OA, HD, etc)
    IndicesOfAtomConnecting: list
      The indices (in a PDB object) of all atoms bonded to this one.
    charge: float
      Associated electrostatic charge.
    resid: int
      The residue number in the receptor (listing the protein as a chain from
      N-Terminus to C-Terminus). Assumes this is a protein atom.
    chain: string
      Chain identifier for molecule. See PDB spec.
    structure: string
      One of ALPHA, BETA, or OTHER for the type of protein secondary
      structure this atom resides in (assuming this is a receptor atom).
    comment: string
      Either LIGAND or RECEPTOR depending on whether this is a ligand or
      receptor atom.
    """
    self.atomname = atomname
    self.residue = residue
    self.coordinates = coordinates
    self.element = element
    self.pdb_index = pdb_index
    self.line = line
    self.atomtype = atomtype
    if indices_of_atoms_connecting is not None:
      self.indices_of_atoms_connecting = indices_of_atoms_connecting
    else:
      self.indices_of_atoms_connecting = []
    self.charge = charge
    self.resid = resid
    self.chain = chain
    self.structure = structure
    self.comment = comment

  def copy_of(self):
    """Make a copy of this atom."""
    theatom = Atom()
    theatom.atomname = self.atomname
    theatom.residue = self.residue
    theatom.coordinates = self.coordinates.copy_of()
    theatom.element = self.element
    theatom.pdb_index = self.pdb_index
    theatom.line = self.line
    theatom.atomtype = self.atomtype
    theatom.indices_of_atoms_connecting = self.indices_of_atoms_connecting[:]
    theatom.charge = self.charge
    theatom.resid = self.resid
    theatom.chain = self.chain
    theatom.structure = self.structure
    theatom.comment = self.comment

    return theatom

  def create_pdb_line(self, index):
    """
    Generates appropriate ATOM line for pdb file.

    Parameters
    ----------
    index: int
      Index in associated PDB file.
    """
    output = "ATOM "
    output = (
        output + str(index).rjust(6) + self.atomname.rjust(5) +
        self.residue.rjust(4) + self.chain.rjust(2) + str(self.resid).rjust(4))
    coords = self.coordinates.as_array()  # [x, y, z]
    output = output + ("%.3f" % coords[0]).rjust(12)
    output = output + ("%.3f" % coords[1]).rjust(8)
    output = output + ("%.3f" % coords[2]).rjust(8)
    output = output + self.element.rjust(24)
    return output

  def number_of_neighbors(self):
    """Reports number of neighboring atoms."""
    return len(self.indices_of_atoms_connecting)

  def add_neighbor_atom_indices(self, indices):
    """
    Adds atoms with provided PDB indices as neighbors.

    Parameters
    ----------
    index: list
      List of indices of neighbors in PDB object.
    """
    for index in indices:
      if index not in self.indices_of_atoms_connecting:
        self.indices_of_atoms_connecting.append(index)

  def side_chain_or_backbone(self):
    """Determine whether receptor atom belongs to residue sidechain or backbone.
    """
    # TODO(rbharath): Should this be an atom function?
    if (self.atomname.strip() == "CA" or self.atomname.strip() == "C" or
        self.atomname.strip() == "O" or self.atomname.strip() == "N"):
      return "BACKBONE"
    else:
      return "SIDECHAIN"

  def read_atom_pdb_line(self, line):
    """
    TODO(rbharath): This method probably belongs in the PDB class, and not
    in the Atom class.

    Reads an ATOM or HETATM line from PDB and instantiates fields.

    Atoms in PDBs are represented by ATOM or HETATM statements. ATOM and
    HETATM statements follow the following record format:

    (see ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_Letter.pdf)

    COLUMNS   DATA TYPE       FIELD             DEFINITION
    -------------------------------------------------------------------------------------
    1 - 6     Record name     "ATOM "/"HETATM"
    7 - 11    Integer         serial            Atom serial number.
    13 - 16   Atom            name              Atom name.
    17        Character       altLoc            Alternate location indicator.
    18 - 20   Residue name    resName           Residue name.
    22        Character       chainID           Chain identifier.
    23 - 26   Integer         resSeq            Residue sequence number.
    27        AChar           iCode             Code for insertion of residues.
    31 - 38   Real(8.3)       x                 Orthogonal coordinates for X in Angstroms.
    39 - 46   Real(8.3)       y                 Orthogonal coordinates for Y in Angstroms.
    47 - 54   Real(8.3)       z                 Orthogonal coordinates for Z in Angstroms.
    55 - 60   Real(6.2)       occupancy         Occupancy.
    61 - 66   Real(6.2)       tempFactor        Temperature factor.
    77 - 78   LString(2)      element           Element symbol, right-justified.
    79 - 80   LString(2)      charge            Charge on the atom.
    """
    self.line = line
    self.atomname = line[11:16].strip()

    if len(self.atomname) == 1:
      self.atomname = self.atomname + "  "
    elif len(self.atomname) == 2:
      self.atomname = self.atomname + " "
    elif len(self.atomname) == 3:
      # This line is necessary for babel to work, though many PDBs in
      # the PDB would have this line commented out
      self.atomname = self.atomname + " "

    self.coordinates = Point(coords=np.array(
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]))

    # now atom type (for pdbqt)
    if line[77:79].strip():
      self.atomtype = line[77:79].strip().upper()
    elif self.atomname:
      # If atomtype is not specified, but atomname is, set atomtype to the
      # first letter of atomname. This heuristic suffices for proteins,
      # since no two-letter elements appear in standard amino acids.
      self.atomtype = self.atomname[:1]
    else:
      self.atomtype = ""

    if line[69:76].strip() != "":
      self.charge = float(line[69:76])
    else:
      self.charge = 0.0

    if self.element == "":  # try to guess at element from name
      two_letters = self.atomname[0:2].strip().upper()
      valid_two_letters = [
          "BR", "CL", "BI", "AS", "AG", "LI", "HG", "MG", "MN", "RH", "ZN", "FE"
      ]
      if two_letters in valid_two_letters:
        self.element = two_letters
      else:  #So, just assume it's the first letter.
        # Any number needs to be removed from the element name
        self.element = self.atomname
        self.element = self.element.replace('0', '')
        self.element = self.element.replace('1', '')
        self.element = self.element.replace('2', '')
        self.element = self.element.replace('3', '')
        self.element = self.element.replace('4', '')
        self.element = self.element.replace('5', '')
        self.element = self.element.replace('6', '')
        self.element = self.element.replace('7', '')
        self.element = self.element.replace('8', '')
        self.element = self.element.replace('9', '')
        self.element = self.element.replace('@', '')

        self.element = self.element[0:1].strip().upper()

    self.pdb_index = line[6:12].strip()
    self.residue = line[16:20]
    # this only uses the rightmost three characters, essentially
    # removing unique rotamer identification
    self.residue = " " + self.residue[-3:]

    if line[23:26].strip() != "":
      self.resid = int(line[23:26])
    else:
      self.resid = 1

    self.chain = line[21:22]
    if self.residue.strip() == "":
      self.residue = " MOL"


class Charged(object):
  """
  A class that represeents a charged atom.
  """

  def __init__(self, coordinates, indices, positive):
    """
    Parameters
    ----------
    coordinates: point
      Coordinates of atom.
    indices: list
      Contains boolean true or false entries for self and neighbors to
      specify if positive or negative charge
    positive: bool
      Whether this atom is positive or negative.
    """
    self.coordinates = coordinates
    self.indices = indices
    self.positive = positive


def vector_subtraction(point1, point2):  # point1 - point2
  """Subtracts the coordinates of the provided points."""
  return Point(coords=point1.as_array() - point2.as_array())


def cross_product(point1, point2):  # never tested
  """Calculates the cross-product of provided points."""
  return Point(coords=np.cross(point1.as_array(), point2.as_array()))


def vector_scalar_multiply(point, scalar):
  """Multiplies the provided point by scalar."""
  return Point(coords=scalar * point.as_array())


def dot_product(point1, point2):
  """Dot product of points."""
  return np.dot(point1.as_array(), point2.as_array())


def dihedral(point1, point2, point3, point4):  # never tested
  """Compute dihedral angle between 4 points.

    TODO(rbharath): Write a nontrivial test for this.
  """

  b1 = vector_subtraction(point2, point1)
  b2 = vector_subtraction(point3, point2)
  b3 = vector_subtraction(point4, point3)

  b2Xb3 = cross_product(b2, b3)
  b1Xb2 = cross_product(b1, b2)

  b1XMagb2 = vector_scalar_multiply(b1, b2.magnitude())
  radians = math.atan2(dot_product(b1XMagb2, b2Xb3), dot_product(b1Xb2, b2Xb3))
  return radians


def angle_between_three_points(point1, point2, point3):
  """Computes the angle (in radians) between the three provided points."""
  return angle_between_points(
      vector_subtraction(point1, point2), vector_subtraction(point3, point2))


def angle_between_points(point1, point2):
  """Computes the angle (in radians) between two points."""
  return math.acos(
      dot_product(point1, point2) / (point1.magnitude() * point2.magnitude()))


def normalized_vector(point):
  """Normalize provided point."""
  return Point(coords=point.as_array() / np.linalg.norm(point.as_array()))


def distance(point1, point2):
  """Computes distance between two points."""
  return point1.dist_to(point2)


def project_point_onto_plane(point, plane_coefficients):
  """Finds nearest point on specified plane to given point.

  Parameters
  ----------
  point: Point
    Given point
  plane_coefficients: list
    [a, b, c, d] where place equation is ax + by + cz = d
  """
  # The normal vector to plane is n = [a, b, c]
  offset = plane_coefficients[3]
  normal = np.array(plane_coefficients[:3])
  # We first shift by basepoint (a point on given plane) to make math
  # simpler. basepoint is given by d/||n||^2 * n
  basepoint = (offset / np.linalg.norm(normal)**2) * normal
  diff = point.as_array() - basepoint
  # The perpendicular component of diff to plane is
  # (n^T diff / ||n||^2) * n
  perp = (np.dot(normal, diff) / np.linalg.norm(normal)**2) * normal
  closest = basepoint + (diff - perp)
  return Point(coords=np.array(closest))
