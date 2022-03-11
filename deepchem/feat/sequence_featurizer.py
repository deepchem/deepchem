from deepchem.feat import OneHotFeaturizer

class PFMFeaturizer(Featurizer):
    #need to make abstract class
    """
    Encodes a probability frequency matrix for a given multiple sequence alignment

    Examples
    --------
    >>> from deepchem.feat import SequenceFeaturizer
    >>> from deepchem.utils.sequence_utils import MSA_to_dataset
    >>> msa_path = hhsearch('deepchem/utils/test/data/example.fasta', database='example_db', data_dir='deepchem/utils/test/data/', evalue=0.001, num_iterations=2, num_threads=4)
    >>> dataset = MSA_to_dataset(msa_path)
    >>> featurizer = PFMFeaturizer()
    >>> pfm = featurizer.featurize(dataset)
    
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
