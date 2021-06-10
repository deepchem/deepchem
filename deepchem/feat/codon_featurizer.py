import logging

logger = logging.getLogger(__name__)

from deepchem.feat.base_classes import Featurizer
from typing import Any

ProteinSeqs = 
RNACodons = {"UUU": 0, "UUC": 1, "UUA": 2, "UUG": 3, "UCU": 4, "UCC": 5, "UCA": 6, "UCG": 7, "UAU": 8, "UAC": 9, "UAA": 10, "UAG": 11, "UGU": 12, "UGC": 13, "UGA": 14, "UGG": 15, "CUU": 16, "CUC": 17, "CUA": 18, "CUG": 19, "CCU": 20, "CCC": 21, "CCA": 22, "CCG": 23, "CAU": 24, "CAC": 25, "CAA": 26, "CAG": 27, "CGU": 28, "CGC": 29, "CGA": 30, "CGG": 31, "AUU": 32, "AUC": 33, "AUA": 34, "AUG": 35, "ACU": 36, "ACC": 37, "ACA": 38, "ACG": 39, "AAU": 40, "AAC": 41, "AAA": 42, "AAG": 43, "AGU": 44, "AGC": 45, "AGA": 46, "AGG": 47, "GUU": 48, "GUC": 49, "GUA": 50, "GUG": 51, "GCU": 52, "GCC": 53, "GCA": 54, "GCG": 55, "GAU": 56, "GAC": 57, "GAA": 58, "GAG": 59, "GGU": 60, "GGC": 61, "GGA": 62, "GGG": 63}


class CodonFeaturizer(Featurizer):
  
