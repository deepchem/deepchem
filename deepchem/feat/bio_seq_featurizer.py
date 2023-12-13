import numpy as np
import deepchem as dc
import pysam


class SAMFeaturizer:

    def __init__(self, max_records=None):
        self.max_records = max_records

    def get_features(self, samfile):
        """
        Extract features from a SAM file.
        Parameters:
        - samfile: SAM file.
        Returns:
        - features: A 2D NumPy array representing the extracted features.
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

        return np.array(features)
