import numpy as np

import logging
logger = logging.getLogger(__name__)

from deepchem.feat.base_classes import Featurizer
from typing import Iterable

FASTA_tokens = {
  "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19, "U":20, "V":21, "W":22, "Y":23, "Z":24, "X":25, "*":26, "-":27
}

class ProteinTokenizer(Featurizer):
  def tokenize(proteins: Iterable[String], log_every_n: int = 1000) -> np.ndarray:
    # Calls featurize() and returns its output
    return featurize(proteins, log_every_n)

  def featurize(self, proteins: Iterable[String], log_every_n: int = 1000) -> np.ndarray:
    # Return empty array if no proteins are provided as input
    if (len(proteins) <= 0):
      return np.ndarray([])
    # Calls featurize() in parent class, which will call _featurize() for each protein in proteins
    return Featurizer.featurize(self, proteins, log_every_n)

  # TODO: Rename
  def _purify(protein: String):
    first_idx = -1
    for i in range(len(protein))
      if (first_idx == -1 and protein[i:i+5] == "[CLS]"):
        first_idx = i+5
      elif (first_idx != -1 and protein[i, i+5] == "[SEP]"):
        return(protein[first_idx:i])

  def _featurize(self, protein: str):
    protein = protein.upper()
    protein = _purify(protein)
    for letter in protein:
      # Check it exists
      # If yes, add token to tuple/array
      # If no, log and skip

  def untransform():
