import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.utils.data_utils import pad_array
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.molecule_featurizers.atomic_coordinates import AtomicCoordinates


class BPSymmetryFunctionInput(MolecularFeaturizer):
  """Calculate symmetry function for each atom in the molecules

  This method is described in [1]_

  References
  ----------
  .. [1] Behler, Jörg, and Michele Parrinello. "Generalized neural-network
     representation of high-dimensional potential-energy surfaces." Physical
     review letters 98.14 (2007): 146401.

  Note
  ----
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
    self.coordfeat = AtomicCoordinates(use_bohr=True)

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """Calculate symmetry function.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      A numpy array of symmetry function. The shape is `(max_atoms, 4)`.
    """
    coordinates = self.coordfeat._featurize(mol)
    atom_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atom_numbers = np.expand_dims(atom_numbers, axis=1)
    assert atom_numbers.shape[0] == coordinates.shape[0]
    features = np.concatenate([atom_numbers, coordinates], axis=1)
    return pad_array(features, (self.max_atoms, 4))
