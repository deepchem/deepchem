from deepchem.feat.base_classes import Featurizer
from typing import Iterable
import numpy as np
import logging
logger = logging.getLogger(__name__)

# Integer tokens for every amino acid code in the FASTA format.
FASTA_tokens = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "Y": 23,
    "Z": 24,
    "X": 25,
    "*": 26,
    "-": 27
}


class ProteinTokenizer(Featurizer):
  """Tokenizes protein sequences in FASTA-format with start and end annotations
  into integer arrays.

  This tokenizer takes in an array of FASTA-format protein sequence strings
  that are terminated on both ends with [CLS] and [SEP] annotations, and
  converts each string into a matching array of integer tokens.

  The tokenizer is case-insensitive and safely ignores all non-FASTA characters
  (spaces, newlines, commas, etc.).

  eg. ["EMV[CLS]ABCDE[SEP]X", "WMY[CLS]FGH[SEP]XAB"] -> [[0, 1, 2, 3, 4], [5, 6, 7]]
  """

  def featurize(self,
                proteins: Iterable[str],
                log_every_n: int = 1000) -> np.ndarray:
    """Tokenizes protein sequences.

    Parameters
    ----------
    proteins: Iterable[str]
      An iterable (list, array, etc.) of FASTA-format protein sequence strings
      eg. ["ABCD[CLS]EFGHIJK[SEP]LMNOP", "[CLS]A[SEP]"]
      Newlines, spaces, and commas in protein sequence strings are safely ignored.
    log_every_n: int, optional (default 1000)
      How many proteins are tokenized every time a tokenization is logged.

    Returns
    -------
    np.ndarray
      An array of arrays of integer tokens (one array for every sequence).
    """
    # Calls featurize() in parent class, which will call _featurize() for each protein in proteins
    return Featurizer.featurize(self, proteins, log_every_n)

  def untransform(self, input_sequences: Iterable[Iterable[int]]) -> tuple:
    """Convert from tokenized arrays back into original string.

    Parameters
    ----------
    input_sequences: Iterable[int]
      Iterable of the token arrays to be untransformed.

    Returns
    -------
    tuple
      Tuple of strings, where each element is a protein sequence.
    """
    acid_codes = list(FASTA_tokens.keys())  # List of all keys in FASTA_tokens
    all_tokens = list(
        FASTA_tokens.values())  # List of all values in FASTA_tokens
    output_acid_codes: tuple = tuple(
    )  # FASTA amino acid codes for values in input
    # Iterating through input_tokens
    for sequence in input_sequences:
      token_codes = "[CLS]"
      for token in sequence:
        idx = all_tokens.index(token)  # Get index of token
        code = acid_codes[idx]  # Get corresponding key (FASTA amino acid code)
        token_codes = token_codes + code
      token_codes = token_codes + "[SEP]"
      output_acid_codes = output_acid_codes + (token_codes,)
    return output_acid_codes

  def _extract_relevant_sequence(self, protein: str) -> str:
    """Extracts the relevant part of a protein sequence between the first
    [CLS] mark and the first [SEP] mark following the first [CLS] mark.

    eg. "AB[CLS]ABCD[SEP]A" -> "ABCD"

    Parameters
    ----------
    protein: str
      A FASTA-format protein sequence with start and end annotations.

    Returns
    -------
    str
      A FASTA-format protein sequence without start and end annotations.
    """
    first_idx = -1  # Will be set to first string index after [CLS]
    for i in range(len(protein) - 4):
      if (first_idx == -1 and protein[i:i + 5] == "[CLS]"):
        first_idx = i + 5
        i = first_idx
      elif (first_idx != -1 and protein[i:i + 5] == "[SEP]"):
        return (protein[first_idx:i])  # Note: i is index of "[" in "[SEP]"
    logger.info(
        f"[CLS] [SEP] pair not found for protein {protein}. Skipping...")
    return ("")

  def _featurize(self, protein: str) -> np.array:
    """Tokenizes a protein sequence

    Parameters
    ----------
      protein: str
        A FASTA-format string representation of a protein sequence.

    Returns
    -------
    str
      An array of integer tokens for the protein sequence.
    """
    protein = protein.upper()
    protein = self._extract_relevant_sequence(protein)
    splitProtein = protein.split()
    tokens = np.array([], dtype=int)  # Array of tokens for the protein sequence
    for acid in splitProtein:  # Loops through amino acid codes in protein sequence
      if acid in FASTA_tokens:  # Tokenize amino acid
        token = FASTA_tokens.get(acid)
        tokens = np.append(tokens, token)
      else:  # Ignore invalid FASTA code
        logger.info(f"Invalid FASTA amino acid code {acid}, skipping...")
    return tokens
