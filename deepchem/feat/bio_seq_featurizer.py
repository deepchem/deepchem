import numpy as np
import deepchem as dc
try:
    import pysam
except ImportError:
    print("Error: Unable to import pysam. Please make sure it is installed.")


class SAMFeaturizer:
    """
    A class for extracting features from a SAM file.

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
    get_features(samfile)
        Extract features from a SAM file.

        Parameters
        ----------
        samfile : str
            SAM file.

        Returns
        -------
        features : numpy.ndarray
            A 2D NumPy array representing the extracted features.
            Each row corresponds to a SAM record, and columns represent different features.
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
        Initialize SAMFeaturizer.

        Parameters
        ----------
        max_records : int or None, optional
            The maximum number of records to extract from the SAM file. If None, all records will be extracted.
        """
        self.max_records = max_records

    def get_features(self, samfile):
        """
        Extract features from a SAM file.

        Parameters
        ----------
        samfile : str
            SAM file.

        Returns
        -------
        features : numpy.ndarray
        A 2D NumPy array representing the extracted features.
        """

        features = []
        record_count = 0

        for record in samfile:
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

        samfile.close()

        return np.array(features, dtype="object")


class BAMFeaturizer:
    """
    A class for extracting features from a BAM file.

    Parameters
    ----------
    max_records : int or None, optional
        The maximum number of records to extract from the BAM file. If None, all records will be extracted.

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

    def get_features(self, bamfile):
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

        for record in bamfile:
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

        bamfile.close()

        return np.array(features, dtype="object")
