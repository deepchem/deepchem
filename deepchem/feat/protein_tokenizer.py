import numpy as np

import logging
logger = logging.getLogger(__name__)

from deepchem.feat.base_classes import Featurizer
from typing import Iterable

FASTA_tokens = {
  "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19, "U":20, "V":21, "W":22, "Y":23, "Z":24, "X":25, "*":26, "-":27
}

class ProteinTokenizer(Featurizer):
  def featurize(self, proteins: Iterable[str], log_every_n: int = 1000) -> np.ndarray:
    # Return empty array if no proteins are provided as input
    if (len(proteins) <= 0): 
      return np.ndarray([])
    # Calls featurize() in parent class, which will call _featurize() for each protein in proteins
    return Featurizer.featurize(self, proteins, log_every_n)

  def untransform(self, input_tokens: Iterable):
    acid_codes = list(FASTA_tokens.keys()) # List of all keys in FASTA_tokens
    all_tokens = list(FASTA_tokens.values()) # List of all values in FASTA_tokens
    output_acid_codes = np.array([]) # FASTA amino acid codes for values in input 
    # Iterating through input_tokens
    for token in input_tokens:
      idx = all_tokens.index(token) # Get index of token
      code = acid_codes[idx] # Get corresponding key (FASTA amino acid code)
      output_acid_codes.append(code)
    return output_acid_codes

  def _purify(self, protein: str) -> str:
    first_idx = -1
    for i in range(len(protein)-4):
      if (first_idx == -1 and protein[i:i+5] == "[CLS]"):
        first_idx = i+5
        i = first_idx
      elif (first_idx != -1 and protein[i:i+5] == "[SEP]"):
        return(protein[first_idx:i])
    logger.info(f"[CLS] [SEP] pair not found for protein {protein}. Skipping...")
    return("")

  def _featurize(self, protein: str) -> np.array:
    protein = protein.upper()
    protein = self._purify(protein)
    protein = protein.split()
    tokens = np.array([])
    for acid in protein:
      if acid in FASTA_tokens:
        token = FASTA_tokens.get(acid)
        tokens = np.append(tokens, token)
      else:
        logger.info(f"Invalid FASTA amino acid code {acid}, skipping...")
    return tokens
