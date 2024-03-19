import numpy as np
from deepchem.feat import Featurizer


class SAMFeaturizer(Featurizer):
    """
    Featurizes SAM files, that store biological sequences aligned to a reference
    sequence. This class extracts Query Name, Query Sequence, Query Length,
    Reference Name,Reference Start, CIGAR and Mapping Quality of each read in
    a SAM file.

    This is the default featurizer used by SAMLoader, and it extracts the following
    fields from each read in each SAM file in the given order:-
    - Column 0: Query Name
    - Column 1: Query Sequence
    - Column 2: Query Length
    - Column 3: Reference Name
    - Column 4: Reference Start
    - Column 5: CIGAR
    - Column 6: Mapping Quality

    Examples
    --------
    >>> from deepchem.data.data_loader import SAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.sam'
    >>> featurizer = dc.feat.SAMFeaturizer()
    >>> features = featurizer.featurize(inputs)
    >>> type(features[0])
    <class 'numpy.ndarray'>

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with Linux or MacOS X.
    To use Pysam on Windows, use Windows Subsystem for Linux(WSL).

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

    def _featurize(self, datapoint):
        """
        Extract features from a SAM file.

        Parameters
        ----------
        datapoint : str
            Name of SAM file.

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


class BAMFeaturizer(Featurizer):
    """
    Featurizes BAM files, that are compressed binary representations of SAM
    (Sequence Alignment Map) files. This class extracts Query Name, Query
    Sequence, Query Length, Reference Name, Reference Start, CIGAR and Mapping
    Quality of the alignment in the BAM file.

    This is the default featurizer used by BAMLoader, and it extracts the following
    fields from each read in each BAM file in the given order:-
    - Column 0: Query Name
    - Column 1: Query Sequence
    - Column 2: Query Length
    - Column 3: Reference Name
    - Column 4: Reference Start
    - Column 5: CIGAR
    - Column 6: Mapping Quality

    Examples
    --------
    >>> from deepchem.data.data_loader import BAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.bam'
    >>> featurizer = dc.feat.BAMFeaturizer()
    >>> features = featurizer.featurize(inputs)
    >>> type(features[0])
    <class 'numpy.ndarray'>

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with Linux or MacOS X.
    To use Pysam on Windows, use Windows Subsystem for Linux(WSL).

    """

    def __init__(self, max_records=None):
        """
        Initialize BAMFeaturizer.

        Parameters
        ----------
        max_records : int or None, optional
            The maximum number of records to extract from the BAM file. If None, all
            records will be extracted.

        """
        self.max_records = max_records

    def _featurize(self, datapoint):
        """
        Extract features from a BAM file.

        Parameters
        ----------
        datapoint : str
            Name of the BAM file.
            The corresponding index file must be in the same directory.

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


class CRAMFeaturizer(Featurizer):
    """
    Featurizes CRAM files, that are compressed columnar file format for storing
    biological sequences aligned to a reference sequence. This class extracts Query Name, Query
    Sequence, Query Length, Reference Name, Reference Start, CIGAR and Mapping
    Quality of the alignment in the CRAM file.

    This is the default featurizer used by CRAMLoader, and it extracts the following
    fields from each read in each CRAM file in the given order:-
    - Column 0: Query Name
    - Column 1: Query Sequence
    - Column 2: Query Length
    - Column 3: Reference Name
    - Column 4: Reference Start
    - Column 5: CIGAR
    - Column 6: Mapping Quality

    Examples
    --------
    >>> from deepchem.data.data_loader import CRAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.cram'
    >>> featurizer = dc.feat.CRAMFeaturizer()
    >>> features = featurizer.featurize(inputs)
    >>> type(features[0])
    <class 'numpy.ndarray'>

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with Linux or MacOS X.
    To use Pysam on Windows, use Windows Subsystem for Linux(WSL).

    """

    def __init__(self, max_records=None):
        """
        Initialize CRAMFeaturizer.

        Parameters
        ----------
        max_records : int or None, optional
            The maximum number of records to extract from the CRAM file. If None, all
            records will be extracted.

        """
        self.max_records = max_records

    def _featurize(self, datapoint):
        """
        Extract features from a CRAM file.

        Parameters
        ----------
        datapoint : str
            Name of the CRAM file.

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
