from deepchem.feat.molecule_featurizers import sparse_matrix_one_hot_featurizer

class PFMFeaturizer(Featurizer):
    """
    Encodes a probability frequency matrix for a given multiple sequence alignment

    Examples
    --------
    >>> from deepchem.feat import PFMFeaturizer
    >>> msa = NumpyDataset(X=['ABC','BCD'], ids=['seq1','seq2'])
    >>> featurizer = PFMFeaturizer()
    >>> pfm = featurizer.featurize(msa)
    >>> pfm.shape[0]
    (4,3)
    
    """
    def __init__(self):
        """Initialize SequenceFeaturizer
        

        """
    def _featurize(self, dataset):
        """Featurize a sequence into a probability frequency matrix

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset to featurize.

        Returns
        -------
        pfm: np.ndarray
            Probability frequency matrix for the set of sequences.    
        """
        one_hot_encoder = OneHotFeaturizer() #sparse matrix one hot featurizer?
        sequences = dataset.X
        sequences_one_hot = one_hot_encoder.featurize(sequences)

        pfm = np.sum(np.array(sequences_one_hot), axis=1)
        for i, res_freq in enumerate(pfm):
            total_count = np.sum(res_freq)
            if total_count > 0:
            # Calculate frequency
            pfm[i, :] = pfm[i, :]/total_count
        return pfm
