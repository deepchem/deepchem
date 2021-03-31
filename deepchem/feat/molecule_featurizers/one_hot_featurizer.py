import logging
from typing import List

import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.feat.base_classes import Featurizer
from deepchem.feat.base_classes import MolecularFeaturizer
from typing import Any, Iterable
from icecream import ic

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

  def featurize(self, datapoints: Iterable[Any], 
                log_every_n: int = 1000) -> np.ndarray:
    datapoints = list(datapoints)
    if (len(datapoints) < 1):
      print("No datapoints are present in the parameter Iterable, so we return an empty array.")
      return np.array([])
    ic(type(datapoints[0]))
    # Featurize Mol data
    if (type(datapoints[0]) == RDKitMol): # Mol
      return MolecularFeaturizer.featurize(self, datapoints, log_every_n)
    # Featurize str data
    elif (type(datapoints[0]) == str): # String
      return Featurizer.featurize(self, datapoints, log_every_n)
    else:
      print("One hot featurizer only supports strings and mols at this time, so returning {}")
      return np.array([])

  def _featurize(self, datapoint: Any):
    ic(datapoint)
    if (type(datapoint)==str):
      return self._featurizeString(datapoint)
    elif (type(datapoint)==RDKitMol):
      return self._featurizeMol(datapoint)
    else:
      print("One hot featurizer only supports strings and mols at this time, so returning {}")
      return np.array([])

  def _featurizeString(self, string: str) -> np.ndarray:
    """Compute one-hot featurization of string.

    Parameters
    ----------
    str: An arbitrary string to be featurized.

    Returns
    -------
    np.ndarray
      An one hot vector encoded from arbitrary input string.
      The shape is `(max_length, len(charset) + 1)`.
      The index of unknown character is `len(charset)`.
    """
    # validation
    if (len(string) > self.max_length):
      logger.info(
          "The length of {} is longer than `max_length`. So we return an empty array."
      )
      return np.array([])

    string = self.pad_string(string)
    ic(string)
    return np.array([
        one_hot_encode(val, self.charset, include_unknown_set=True)
        for val in string
    ])

  def _featurizeMol(self, mol: RDKitMol) -> np.ndarray:
    """Compute one-hot featurization of this molecule.

    Parameters
    ----------
    mol: rdKit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      An one hot vector encoded from SMILES.
      The shape is '(max_length, len(charset) + 1)'
      The index of unknown character is 'len(charset)'.
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")
    smiles = Chem.MolToSmiles(mol)
    return self._featurize(smiles)

  def pad_string(self, string: str) -> str:
    """Pad SMILES string to `self.pad_length`

    Parameters
    ----------
    string: str
      The string to be padded.

    Returns
    -------
    str
      String space padded to self.pad_length
    """
    return string.ljust(self.max_length)

  def untransform(self, one_hot_vectors: np.ndarray) -> str:
    """Convert from one hot representation back to original string

    Parameters
    ----------
    one_hot_vectors: np.ndarray
      An array of one hot encoded features.

    Returns
    -------
    str
      Original string for an one hot encoded array.
    """
    string = ""
    for one_hot in one_hot_vectors:
      try:
        idx = np.argmax(one_hot)
        string += self.charset[idx]
      except IndexError:
        string += ""
    return string
