"""
Genomic data handling utilities.
"""
import simdna
from simdna.synthetic import LoadedEncodeMotifs

loaded_motifs = LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH,
                                   pseudocountProb=0.001)

