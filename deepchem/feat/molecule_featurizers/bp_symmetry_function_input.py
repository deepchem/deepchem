import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.utils.data_utils import pad_array
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.molecule_featurizers.atomic_coordinates import AtomicCoordinates


class BPSymmetryFunctionInput(MolecularFeaturizer):
  """Calculate symmetry function for each atom in the molecules

  This method is described in [1]_.

  Examples
  --------
  >>> import deepchem as dc
  >>> smiles = ['C1C=CC=CC=1']
  >>> featurizer = dc.feat.BPSymmetryFunctionInput(max_atoms=10)
  >>> features = featurizer.featurize(smiles)
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape  # (max_atoms, 4)
  (10, 4)

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

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
    """Calculate symmetry function.

    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      A numpy array of symmetry function. The shape is `(max_atoms, 4)`.
    """
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )
    coordinates = self.coordfeat._featurize(datapoint)
    atom_numbers = np.array(
        [atom.GetAtomicNum() for atom in datapoint.GetAtoms()])
    atom_numbers = np.expand_dims(atom_numbers, axis=1)
    assert atom_numbers.shape[0] == coordinates.shape[0]
    features = np.concatenate([atom_numbers, coordinates], axis=1)
    return pad_array(features, (self.max_atoms, 4))
