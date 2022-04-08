import numpy as np
# from deepchem.feat.molecule_featurizers import SparseMatrixOneHotFeaturizer
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.feat.base_classes import Featurizer
from typing import List, Optional

CHARSET = [ #centralize charset somewhere? Import from SparseMatrixOneHotFeaturizer?
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', 'X', 'Z', 'U', 'O'
]

class PFMFeaturizer(Featurizer):
    """
    Encodes a list probability frequency matrices for a given list of multiple sequence alignments

    Examples
    --------
    >>> from deepchem.feat.sequence_featurizers import PFMFeaturizer
    >>> msa = NumpyDataset(X=['ABC','BCD'], ids=['seq1','seq2'])
    >>> seqs = msa.X
    >>> featurizer = PFMFeaturizer()
    >>> pfm = featurizer.featurize(seqs)
    >>> pfm.shape[0]
    (25,3)
    
    """
    def __init__(self,
               charset: List[str] = CHARSET): #default charset for amino acids? Make required argument?
               #max_length? 25? 100?
        """Initialize featurizer.

        Parameters
        ----------
        charset: List[str] (default code)
            A list of strings, where each string is length 1 and unique.
        max_length: int, optional (default 25)
            Maximum length of sequences to be featurized.
        """
        if len(charset) != len(set(charset)):
            raise ValueError("All values in charset must be unique.")
        self.charset = charset
        # self.max_length = max_length
        self.ohe = OneHotFeaturizer(charset = CHARSET)

    def _featurize(self, datapoint): #datapoint is entire msa, not a single sequence
        """Featurize a multisequence alignment into a probability frequency matrix

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset to featurize.

        Returns
        -------
        pfm: np.ndarray
            Probability frequency matrix for the set of sequences.    
        """
        # one_hot_encoder = OneHotFeaturizer() #self.ohe
        # seq_one_hot = self.ohe.featurize(datapoint)
        seqs_one_hot = np.array([one_hot_encode(seq, self.charset, include_unknown_set=False) for seq in datapoint])
        # PNET: sequences_one_hot = [[to_one_hot(sequence[i]) for i in range(len(sequence))] for sequence in sequences]
        #need entire dataset encoded before making pfm
        pfm = np.sum(np.array(seqs_one_hot), axis=1)
        for i, res_freq in enumerate(pfm):
            total_count = np.sum(res_freq)
            if total_count > 0:
            # Calculate frequency
                pfm[i, :] = pfm[i, :]/total_count
        return pfm
