import numpy as np
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
from deepchem.feat.base_classes import Featurizer
from typing import List, Optional

CHARSET = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', 'X', 'Z', 'B', 'U', 'O'
]


class PFMFeaturizer(Featurizer):
    """
    Encodes a list position frequency matrices for a given list of multiple sequence alignments

    The default character set is 25 amino acids. If you want to use a different character set, such as nucleotides, simply pass in
    a list of character strings in the featurizer constructor.

    The max_length parameter is the maximum length of the sequences to be featurized. If you want to featurize longer sequences, modify the
    max_length parameter in the featurizer constructor.

    The final row in the position frequency matrix is the unknown set, if there are any characters which are not included in the charset.

    Examples
    --------
    >>> from deepchem.feat.sequence_featurizers import PFMFeaturizer
    >>> from deepchem.data import NumpyDataset
    >>> msa = NumpyDataset(X=[['ABC','BCD'],['AAA','AAB']], ids=[['seq01','seq02'],['seq11','seq12']])
    >>> seqs = msa.X
    >>> featurizer = PFMFeaturizer()
    >>> pfm = featurizer.featurize(seqs)
    >>> pfm.shape
    (2, 26, 100)

    """

    def __init__(self,
                 charset: List[str] = CHARSET,
                 max_length: Optional[int] = 100):
        """Initialize featurizer.

        Parameters
        ----------
        charset: List[str] (default CHARSET)
            A list of strings, where each string is length 1 and unique.
        max_length: int, optional (default 25)
            Maximum length of sequences to be featurized.
        """
        if len(charset) != len(set(charset)):
            raise ValueError("All values in charset must be unique.")
        self.charset = charset
        self.max_length = max_length
        self.ohe = OneHotFeaturizer(charset=CHARSET, max_length=max_length)

    def _featurize(self, datapoint):
        """Featurize a multisequence alignment into a position frequency matrix

        Use dc.utils.sequence_utils.hhblits or dc.utils.sequence_utils.hhsearch to create a multiple sequence alignment from a fasta file.

        Parameters
        ----------
        datapoint: np.ndarray
            MSA to featurize. A list of sequences which have been aligned and padded to the same length.

        Returns
        -------
        pfm: np.ndarray
            Position frequency matrix for the set of sequences with the rows corresponding to the unique characters and the columns corresponding to the position in the alignment.

        """

        seq_one_hot = self.ohe.featurize(datapoint)

        seq_one_hot_array = np.transpose(
            np.array(seq_one_hot), (0, 2, 1)
        )  # swap rows and columns to make rows the characters, columns the positions

        pfm = np.sum(seq_one_hot_array, axis=0)

        return pfm


def PFM_to_PPM(pfm):
    """
    Calculate position probability matrix from a position frequency matrix
    """
    ppm = pfm.copy()
    for col in range(ppm.shape[1]):
        total_count = np.sum(ppm[:, col])
        if total_count > 0:
            # Calculate frequency
            ppm[:, col] = ppm[:, col] / total_count
    return ppm
