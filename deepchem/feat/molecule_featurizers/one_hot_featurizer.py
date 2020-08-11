import numpy as np
from typing import List

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer

ZINC_CHARSET = [
    ' ', '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8',
    '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l',
    'o', 'n', 'p', 's', 'r'
]


class OneHotFeaturizer(MolecularFeaturizer):
  """Encodes a molecule as a one-hot array.

  This featurizer takes a molecule and encodes its Smiles string as a one-hot
  array.

  Notes
  -----
  This class requires RDKit to be installed.
  Note that this featurizer is not thread Safe in initialization of charset
  """

  def __init__(self, charset: List[str] = ZINC_CHARSET, padlength: int = 120):
    """Initialize featurizer.

    Parameters
    ----------
    charset: List[str]
      A list of strings, where each string is length 1.
    padlength: int, optional (default 120)
      length to pad the smile strings to.
    """
    self.charset = charset
    self.pad_length = padlength

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """Compute one-hot featurization of this molecule.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      Vector of RDKit descriptors for `mol`
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")

    smiles = Chem.MolToSmiles(mol)
    if self.charset is None:
      self.charset = self._create_charset(smiles)
    return np.array([self.one_hot_encoded(smile) for smile in smiles])

  def one_hot_array(self, i: int) -> List[int]:
    """Create a one hot array with bit i set to 1

    Parameters
    ----------
    i: int
      bit to set to 1

    Returns
    -------
    List[int]
      The one hot list of bit i. The length is len(self.charset)
    """
    return [int(x) for x in [ix == i for ix in range(len(self.charset))]]

  def one_hot_index(self, c: str) -> int:
    """Compute one-hot index of charater.

    Parameters
    ----------
    c: str
      character whose index we want

    Returns
    -------
    int
      index of c in self.charset
    """
    return self.charset.index(c)

  def pad_smile(self, smile: str) -> str:
    """Pad a smile string to `self.pad_length`

    Parameters
    ----------
    smile: str
      The smiles string to be padded.

    Returns
    -------
    str
      smile string space padded to self.pad_length
    """

    return smile.ljust(self.pad_length)

  def one_hot_encoded(self, smile: str) -> np.ndarray:
    """One Hot Encode an entire SMILE string

    Parameters
    ----------
    smile: str
      smile string to encode

    Returns
    -------
    np.ndarray
      The one hot encoded arrays for each character in smile
    """
    return np.array([
        self.one_hot_array(self.one_hot_index(x)) for x in self.pad_smile(smile)
    ])

  def untransform(self, one_hot: np.ndarray) -> List[str]:
    """Convert from one hot representation back to SMILE

    Parameters
    ----------
    z: np.ndarray
      A numpy array of one hot encoded features

    Returns
    -------
    List[str]
      The List smile Strings picking MAX for each one hot encoded array
    """
    smiles_list = []
    for i in range(len(one_hot)):
      smiles = ""
      for j in range(len(one_hot[i])):
        char_bit = np.argmax(one_hot[i][j])
        smiles += self.charset[char_bit]
      smiles_list.append(smiles.strip())
    return smiles_list

  def _create_charset(self, smiles: List[str]) -> List[str]:
    """Create the charset from smiles

    Parameters
    ----------
    smiles: List[str]
      List of smile strings

    Returns
    -------
    List[str]
      List of length one strings that are characters in smiles. No duplicates
    """
    s = set()
    for smile in smiles:
      for c in smile:
        s.add(c)
    return [' '] + sorted(list(s))
