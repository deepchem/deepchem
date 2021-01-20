"""
Featurizer implementations used in Smiles2Vec models.
SmilesToSeq featurizer for Smiles2Vec models taken from https://arxiv.org/abs/1712.02734
"""
from typing import Dict, List
import numpy as np
import pandas as pd

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer

PAD_TOKEN = "<pad>"
OUT_OF_VOCAB_TOKEN = "<unk>"


def create_char_to_idx(filename: str,
                       max_len: int = 250,
                       smiles_field: str = "smiles") -> Dict[str, int]:
  """Creates a dictionary with character to index mapping.

  Parameters
  ----------
  filename: str
      Name of the file containing the SMILES strings
  max_len: int, default 250
      Maximum allowed length of the SMILES string
  smiles_field: str, default "smiles"
      Field indicating the SMILES strings int the file.

  Returns
  -------
  Dict[str, int]
    A dictionary mapping characters to their integer indexes.
  """
  smiles_df = pd.read_csv(filename)
  char_set = set()
  for smile in smiles_df[smiles_field]:
    if len(smile) <= max_len:
      char_set.update(set(smile))

  unique_char_list = list(char_set)
  unique_char_list += [PAD_TOKEN, OUT_OF_VOCAB_TOKEN]
  char_to_idx = {letter: idx for idx, letter in enumerate(unique_char_list)}
  return char_to_idx


class SmilesToSeq(MolecularFeaturizer):
  """
  SmilesToSeq Featurizer takes a SMILES string, and turns it into a sequence.
  Details taken from [1]_.

  SMILES strings smaller than a specified max length (max_len) are padded using
  the PAD token while those larger than the max length are not considered. Based
  on the paper, there is also the option to add extra padding (pad_len) on both
  sides of the string after length normalization. Using a character to index (char_to_idx)
  mapping, the SMILES characters are turned into indices and the
  resulting sequence of indices serves as the input for an embedding layer.

  References
  ----------
  .. [1] Goh, Garrett B., et al. "Using rule-based labels for weak supervised
     learning: a ChemNet for transferable chemical property prediction."
     Proceedings of the 24th ACM SIGKDD International Conference on Knowledge
     Discovery & Data Mining. 2018.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self,
               char_to_idx: Dict[str, int],
               max_len: int = 250,
               pad_len: int = 10):
    """Initialize this class.

    Parameters
    ----------
    char_to_idx: Dict
      Dictionary containing character to index mappings for unique characters
    max_len: int, default 250
      Maximum allowed length of the SMILES string.
    pad_len: int, default 10
      Amount of padding to add on either side of the SMILES seq
    """
    self.max_len = max_len
    self.char_to_idx = char_to_idx
    self.idx_to_char = {idx: letter for letter, idx in self.char_to_idx.items()}
    self.pad_len = pad_len

  def to_seq(self, smile: List[str]) -> np.ndarray:
    """Turns list of smiles characters into array of indices"""
    out_of_vocab_idx = self.char_to_idx[OUT_OF_VOCAB_TOKEN]
    seq = [
        self.char_to_idx.get(character, out_of_vocab_idx) for character in smile
    ]
    return np.array(seq)

  def remove_pad(self, characters: List[str]) -> List[str]:
    """Removes PAD_TOKEN from the character list."""
    characters = characters[self.pad_len:]
    characters = characters[:-self.pad_len]
    chars = list()

    for char in characters:
      if char != PAD_TOKEN:
        chars.append(char)
    return chars

  def smiles_from_seq(self, seq: List[int]) -> str:
    """Reconstructs SMILES string from sequence."""
    characters = [self.idx_to_char[i] for i in seq]

    characters = self.remove_pad(characters)
    smile = "".join([letter for letter in characters])
    return smile

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """Featurizes a SMILES sequence.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      A 1D array of a SMILES sequence.
      If the length of SMILES is longer than `max_len`, this value is an empty array.
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    smile = Chem.MolToSmiles(mol)
    if len(smile) > self.max_len:
      return np.array([])

    smile_list = list(smile)
    # Extend shorter strings with padding
    if len(smile) < self.max_len:
      smile_list.extend([PAD_TOKEN] * (self.max_len - len(smile)))

    # Padding before and after
    smile_list += [PAD_TOKEN] * self.pad_len
    smile_list = [PAD_TOKEN] * self.pad_len + smile_list

    smile_seq = self.to_seq(smile_list)
    return smile_seq
