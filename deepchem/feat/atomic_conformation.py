import numpy as np
from deepchem.feat.base_classes import Featurizer
from deepchem.utils import get_partial_charge

from typing import Sequence


class AtomicConformation:
  """This class represents a collection of atoms arranged in 3D space.

  An instance of this class may represent any collection of atoms: a molecule,
  a fragment of a molecule, multiple interacting molecules, a material, etc.
  For each atom it stores a position and a list of scalar properties.  Arbitrary
  properties are supported, but convenience methods are provided for accessing
  certain standard ones: atomic number, formal charge, and partial charge.

  Instances of this class are most often created by AtomicConformationFeaturizer.

  Attributes
  ----------
  positions: ndarray
    the positions of all atoms in Angstroms, stored in an array of shape (N, 3)
    where N is the number of atoms
  properties: ndarray
    the property values for all atoms, stored in an array of shape (N, M) where
    N is the number of atoms and M is the number of properties
  property_names: ndarray
    an array of length M with the names of the properties
  """

  def __init__(self, positions: np.ndarray, properties: np.ndarray,
               property_names: Sequence[str]):
    """Create an AtomicConformation for a set of atoms.

    Parameters
    ----------
    positions: ndarray
      the positions of all atoms in Angstroms, stored in an array of shape (N, 3)
      where N is the number of atoms
    properties: ndarray
      the property values for all atoms, stored in an array of shape (N, M) where
      N is the number of atoms and M is the number of properties
    property_names: Sequence[str]
      the names of the properties
    """
    self.positions = positions
    self.properties = properties
    self.property_names = np.array(property_names)

  @property
  def num_atoms(self) -> int:
    """Get the number of atoms in this object."""
    return self.positions.shape[0]

  def get_property(self, name: str) -> np.ndarray:
    """Get a column of the properties array corresponding to a particular property.

    If there is no property with the specified name, this raises a ValueError.

    Parameters
    ----------
    name: str
      the name of the property to get

    Returns
    -------
    a numpy array containing the requested column of the properties array.  This
    is a 1D array of length num_atoms.
    """
    indices = np.where(self.property_names == name)[0]
    if len(indices) == 0:
      raise ValueError("No property called '%s'" % name)
    return self.properties[:, indices[0]]

  @property
  def atomic_number(self) -> np.ndarray:
    """Get the column of the properties array containing atomic numbers.

    If there is no property with the name 'atomic number', this raises a ValueError.

    Returns
    -------
    a numpy array containing the requested column of the properties array.  This
    is a 1D array of length num_atoms.
    """
    return self.get_property('atomic number')

  @property
  def formal_charge(self) -> np.ndarray:
    """Get the column of the properties array containing formal charges.

    If there is no property with the name 'formal charge', this raises a ValueError.

    Returns
    -------
    a numpy array containing the requested column of the properties array.  This
    is a 1D array of length num_atoms.
    """
    return self.get_property('formal charge')

  @property
  def partial_charge(self) -> np.ndarray:
    """Get the column of the properties array containing partial charges.

    If there is no property with the name 'partial charge', this raises a ValueError.

    Returns
    -------
    a numpy array containing the requested column of the properties array.  This
    is a 1D array of length num_atoms.
    """
    return self.get_property('partial charge')


class AtomicConformationFeaturizer(Featurizer):
  """This featurizer represents each sample as an AtomicConformation object,
  representing a 3D arrangement of atoms.

  It expects each datapoint to be a string, which may be either a filename or a
  SMILES string.  It is processed as follows.

  If the string ends in .pdb, .sdf, or .mol2, it is assumed to be a file in the
  corresponding format.  The positions and elements of all atoms contained in
  the file are loaded.  RDKit is used to compute formal and partial charges.

  Otherwise, it is assumed to be a SMILES string.  RDKit is used to generate a
  3D conformation and to compute formal and partial charges.

  Examples
  --------
  >>> import deepchem as dc
  >>> smiles = ['CCC']
  >>> featurizer = dc.feat.AtomicConformationFeaturizer()
  >>> features = featurizer.featurize(smiles)
  >>> features[0].num_atoms
  11
  >>> sum(features[0].atomic_number == 6)
  3
  >>> sum(features[0].atomic_number == 1)
  8
  >>> type(features[0].formal_charge)
  <class 'numpy.ndarray'>
  >>> features[0].formal_charge.shape
  (11,)
  >>> type(features[0].partial_charge)
  <class 'numpy.ndarray'>
  >>> features[0].partial_charge.shape
  (11,)

  """

  def _featurize(self, datapoint: str, **kwargs) -> AtomicConformation:
    """Calculate features for a single datapoint.

    Parameters
    ----------
    datapoint: str
      This is expected to be either a filename or a SMILES string.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    if datapoint.endswith('.pdb'):
      mols = [Chem.MolFromPDBFile(datapoint, removeHs=False)]
    elif datapoint.endswith('.sdf'):
      supplier = Chem.SDMolSupplier(datapoint, removeHs=False)
      mols = [mol for mol in supplier]
    elif datapoint.endswith('.mol2'):
      mols = [Chem.MolFromMol2File(datapoint, removeHs=False)]
    else:
      mol = Chem.MolFromSmiles(datapoint)
      # SMILES is unique, so set a canonical order of atoms
      new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
      mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
      # Add hydrogens and generate a conformation.
      mol = Chem.AddHs(mol)
      AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
      mols = [mol]

    # Record properties of the molecules.

    positions = []
    properties = []
    for mol in mols:
      positions.append(mol.GetConformer(0).GetPositions())
      AllChem.ComputeGasteigerCharges(mol)
      for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        partial_charge = get_partial_charge(atom)
        properties.append([atomic_num, formal_charge, partial_charge])

    # Create the output object.

    names = ['atomic number', 'formal charge', 'partial charge']
    return AtomicConformation(
        np.concatenate(positions).astype(np.float32),
        np.array(properties, dtype=np.float32), names)
