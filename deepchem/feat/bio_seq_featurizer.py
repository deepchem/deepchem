import numpy as np
import deepchem as dc
try:
    import pysam
except ImportError:
    print("Error: Unable to import pysam. Please make sure it is installed.")

class BAMFeaturizer:
    """
    This class extracts Query Name, Query Sequence, Query Length, Reference Name, 
    Reference Start, CIGAR and Mapping Quality of the alignment in the BAM file.

    Examples
    --------
    >>> from deepchem.data.data_loader import BAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.bam'
    >>> featurizer = dc.feat.BAMFeaturizer()
    >>> features = featurizer.featurize(inputs)
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0][0]     # Query Name
    EASX:X:X:X:X:33:33_1:Y:0:NNNNNN
    >>> features[0][1]     # Query Sequence
    TTTTAATGACGTGCCCGAACTGTGGTTGTATGTTCGTATTGGGATGTAGGTCTTATTGTTGTGTTGTGCAGAAAC
    >>> features[0][2]     # Query Length
    75
    >>> features[0][3]     # Reference Name
    chr1
    >>> features[0][4]     # Reference Start
    2
    >>> features[0][5]     # CIGAR
    [(0, 75)]
    >>> features[0][6]     # Mapping Quality
    42

    Note
    ----
    This class requires pysam to be installed.
    Parameters
    ----------
    max_records : int or None, optional
        The maximum number of records to extract from the SAM file. If None, all records will be extracted.
    Attributes
    ----------
    max_records : int or None
        The maximum number of records to extract.
    Methods
    -------
    get_features(bamfile)
        Extract features from a BAM file.
        Parameters
        ----------
        bamfile : str
            BAM file.
        Returns
        -------
        features : numpy.ndarray
            A 2D NumPy array representing the extracted features.
            Each row corresponds to a BAM record, and columns represent different features.
            - Column 0: Query Name
            - Column 1: Query Sequence
            - Column 2: Query Length
            - Column 3: Reference Name
            - Column 4: Reference Start
            - Column 5: CIGAR
            - Column 6: Mapping Quality
    """

    def __init__(self, max_records=None):
        """
        Initialize BAMFeaturizer.

        Parameters
        ----------
        max_records : int or None, optional
            The maximum number of records to extract from the BAM file. If None, all records will be extracted.
        """
        self.max_records = max_records

    def _featurize(self, datapoint):
        """
        Extract features from a BAM file.

        Parameters
        ----------
        bamfile : str
            BAM file.

        Returns
        -------
        features : numpy.ndarray
        A 2D NumPy array representing the extracted features.
        """

        features = []
        record_count = 0

        for record in datapoint:
            feature_vector = [
                record.query_name,
                record.query_sequence,
                record.query_length,
                record.reference_name,
                record.reference_start,
                record.cigar,
                record.mapping_quality,
            ]

            features.append(feature_vector)
            record_count += 1

            # Break the loop if max_records is set
            if self.max_records is not None and record_count >= self.max_records:
                break

        datapoint.close()

        return np.array(features, dtype="object")
