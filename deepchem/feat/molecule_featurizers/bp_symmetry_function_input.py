import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.atomic_coordinates import AtomicCoordinates


class BPSymmetryFunctionInput(MolecularFeaturizer):
  """Calculate Symmetry Function for each atom in the molecules

  This method is described in [1]_

  References
  ----------
  .. [1] Behler, Jörg, and Michele Parrinello. "Generalized neural-network
     representation of high-dimensional potential-energy surfaces." Physical
     review letters 98.14 (2007): 146401.

  Notes
  -----
  This class requires RDKit to be installed.
  """

  def __init__(self, max_atoms: int):
    """Initialize this featurizer.

    Parameters
    ----------
    max_atoms: int
      The maximum number of atoms expected for molecules this featurizer will
      process.
    """
    self.max_atoms = max_atoms

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    coordfeat = AtomicCoordinates()
    coordinates = coordfeat._featurize(mol)[0]
    atom_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atom_numbers = np.expand_dims(atom_numbers, axis=1)
    assert atom_numbers.shape[0] == coordinates.shape[0]
    n_atoms = atom_numbers.shape[0]
    features = np.concatenate([atom_numbers, coordinates], axis=1)
    return np.pad(features, ((0, self.max_atoms - n_atoms), (0, 0)), 'constant')
