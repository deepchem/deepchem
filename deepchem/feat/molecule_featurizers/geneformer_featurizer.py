import pickle
import logging
import numpy as np
from typing import List
from deepchem.feat.base_classes import Featurizer

logger = logging.getLogger(__name__)


class GeneformerFeaturizer(Featurizer):
    """
    Rank-Value Encoding Featurizer for Geneformer.
    """

    def __init__(self,
                 gene_median_file: str,
                 token_dictionary_file: str,
                 gene_names: List[str],
                 max_length: int = 2048,
                 model_version: str = "V1",
                 padding_token_id: int = 0) -> None:

        self.max_length = max_length
        self.model_version = model_version
        self.gene_names = gene_names
        self.padding_token_id = padding_token_id

        # Load dictionaries
        with open(token_dictionary_file, "rb") as f:
            self.token_dictionary = pickle.load(f)

        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        self.cls_token_id = self.token_dictionary.get("<cls>")
        self.eos_token_id = self.token_dictionary.get("<eos>")

    def _featurize(self, datapoint: np.ndarray) -> np.ndarray:
        # 1. Basic Cleaning
        if not isinstance(datapoint, np.ndarray):
            datapoint = np.array(datapoint)
        if datapoint.ndim > 1:
            datapoint = datapoint.flatten()

        # 2. Total Counts (Library Size)
        total_counts = np.sum(datapoint)
        if total_counts == 0:
            return np.full(self.max_length,
                           self.padding_token_id,
                           dtype=np.int64)

        # 3. Rank-Value Encoding Logic
        nonzero_indices = np.flatnonzero(datapoint)
        nonzero_values = datapoint[nonzero_indices]

        token_ids = []
        normalized_scores = []

        for idx, raw_val in zip(nonzero_indices, nonzero_values):
            if idx < len(self.gene_names):
                gene_name = self.gene_names[idx]

                # Check if gene exists in our dictionaries
                if gene_name in self.token_dictionary and gene_name in self.gene_median_dict:
                    token_ids.append(self.token_dictionary[gene_name])

                    median = self.gene_median_dict[gene_name]
                    if median > 0:
                        # The Core Formula: (Count / Total) / Median
                        norm_val = (raw_val / total_counts) / median
                    else:
                        norm_val = 0
                    normalized_scores.append(norm_val)

        if not token_ids:
            return np.full(self.max_length,
                           self.padding_token_id,
                           dtype=np.int64)

        # 4. Sort Descending by Normalized Score
        sorted_indices = np.argsort(-np.array(normalized_scores))
        ranked_tokens = np.array(token_ids)[sorted_indices]

        # 5. V2 Special Tokens (<cls>, <eos>)
        if self.model_version == "V2" and self.cls_token_id is not None:
            ranked_tokens = np.concatenate(
                ([self.cls_token_id], ranked_tokens, [self.eos_token_id]))

        # 6. Truncate (CRITICAL STEP)
        # This prevents the crash when genes > max_length
        if len(ranked_tokens) > self.max_length:
            ranked_tokens = ranked_tokens[:self.max_length]

        # 7. Pad to max_length
        output = np.full(self.max_length, self.padding_token_id, dtype=np.int64)
        output[:len(ranked_tokens)] = ranked_tokens

        return output
