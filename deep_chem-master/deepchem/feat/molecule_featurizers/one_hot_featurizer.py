import logging
from typing import List

import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.feat.base_classes import MolecularFeaturizer

logger = logging.getLogger(__name__)

ZINC_CHARSET = [
    '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=',
    '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o',
    'n', 'p', 's', 'r'
]


class OneHotFeaturizer(MolecularFeaturizer):
  """Encodes SMILES as a one-hot array.

  This featurizer encodes SMILES string as a one-hot array.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, charset: List[str] = ZINC_CHARSET, max_length: int = 100):
    """Initialize featurizer.

    Parameters
    ----------
    charset: List[str], optional (default ZINC_CHARSET)
      A list of strings, where each string is length 1 and unique.
    max_length: int, optional (default 100)
      The max length for SMILES string. If the length of SMILES string is
      shorter than max_length, the SMILES is padded using space.
    """
    if len(charset) != len(set(charset)):
      raise ValueError("All values in charset must be unique.")
    self.charset = charset
    self.max_length = max_length

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """Compute one-hot featurization of this molecule.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      An one hot vector encoded from SMILES.
      The shape is `(max_length, len(charset) + 1)`.
      The index of unknown character is `len(charset)`.
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    smiles = Chem.MolToSmiles(mol)
    # validation
    if len(smiles) > self.max_length:
      logger.info(
          "The length of {} is longer than `max_length`. So we return an empty array."
      )
      return np.array([])

    smiles = self.pad_smile(smiles)
    return np.array([
        one_hot_encode(val, self.charset, include_unknown_set=True)
        for val in smiles
    ])

  def pad_smile(self, smiles: str) -> str:
    """Pad SMILES string to `self.pad_length`

    Parameters
    ----------
    smiles: str
      The smiles string to be padded.

    Returns
    -------
    str
      SMILES string space padded to self.pad_length
    """
    return smiles.ljust(self.max_length)

  def untransform(self, one_hot_vectors: np.ndarray) -> str:
    """Convert from one hot representation back to SMILES

    Parameters
    ----------
    one_hot_vectors: np.ndarray
      An array of one hot encoded features.

    Returns
    -------
    str
      SMILES string for an one hot encoded array.
    """
    smiles = ""
    for one_hot in one_hot_vectors:
      try:
        idx = np.argmax(one_hot)
        smiles += self.charset[idx]
      except IndexError:
        smiles += ""
    return smiles
