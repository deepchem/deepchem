"""
Geneformer Featurizer for single-cell gene expression data.
"""
import pickle
from typing import Dict, List, Optional

import numpy as np

from deepchem.feat.base_classes import Featurizer


class GeneformerFeaturizer(Featurizer):
    """Rank-Value Encoding featurizer with Vocabulary Mapping.

    This featurizer transforms raw gene expression count vectors into
    token ID sequences using Rank-Value Encoding. The encoding process
    prioritizes genes by their expression levels, enabling transformer
    models to learn from the relative importance of genes within each cell.

    The encoding algorithm:
    1. Maps local gene indices to Global Token IDs (via optional dictionary).
    2. Filters out genes with zero expression.
    3. Sorts remaining tokens by expression value (descending).
    4. Truncates or pads the sequence to a fixed length.

    Parameters
    ----------
    max_length : int, default 2048
        Maximum sequence length for the output token array. Sequences shorter
        than this will be padded; longer sequences will be truncated.
    padding_token : int, default 0
        Token ID used for padding shorter sequences.
    token_dictionary_file : str, optional
        Path to the pickle file containing the gene-to-token dictionary
        (Ensembl ID -> Integer). This is REQUIRED if using pre-trained weights.
        If None, raw column indices are used as token IDs (only for training from scratch).
    gene_names : List[str], optional
        A list of gene names (Ensembl IDs) corresponding to the columns
        of your input array. Required if ``token_dictionary_file`` is provided.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.feat.molecule_featurizers.geneformer_featurizer import GeneformerFeaturizer
    >>> # Scenario 1: Training from scratch (No dictionary)
    >>> featurizer = GeneformerFeaturizer(max_length=5)
    >>> expression = np.array([0, 10, 50, 0, 5]) # Gene 2 is highest (50), Gene 1 (10), Gene 4 (5)
    >>> featurizer.featurize([expression])
    array([[2, 1, 4, 0, 0]])

    Notes
    -----
    The gene indices in the output correspond to token IDs in the Geneformer
    vocabulary. If using pre-trained Geneformer models, you **must** provide
    the ``token_dictionary_file`` and ``gene_names`` to ensure your data
    aligns with the model's vocabulary.

    References
    ----------
    .. [1] Theodoris, C.V., et al. "Transfer learning enables predictions
           in network biology." Nature (2023).
    """

    def __init__(self,
                 max_length: int = 2048,
                 padding_token: int = 0,
                 token_dictionary_file: Optional[str] = None,
                 gene_names: Optional[List[str]] = None) -> None:
        self.max_length = max_length
        self.padding_token = padding_token
        self.gene_names = gene_names
        self.token_dictionary: Optional[Dict[str, int]] = None

        if token_dictionary_file is not None:
            # We allow the FileNotFoundError to propagate up if the file is missing
            # This is better than logging a warning and continuing silently
            with open(token_dictionary_file, "rb") as f:
                self.token_dictionary = pickle.load(f)

            if self.gene_names is None:
                raise ValueError(
                    "If providing a token dictionary, you must also provide 'gene_names' "
                    "so we know which column corresponds to which gene."
                )

    def _featurize(self, datapoint: np.ndarray) -> np.ndarray:
        """Apply Rank-Value Encoding to a single gene expression vector.

        Parameters
        ----------
        datapoint : np.ndarray
            A 1D numpy array of gene expression counts.

        Returns
        -------
        np.ndarray
            A 1D numpy array of shape (max_length,) containing integer token IDs.
        """
        # Ensure input is a numpy array and 1D
        if not isinstance(datapoint, np.ndarray):
            datapoint = np.array(datapoint)
        if datapoint.ndim > 1:
            datapoint = datapoint.flatten()

        # 1. Identify expressed genes
        nonzero_indices = np.flatnonzero(datapoint)
        
        if len(nonzero_indices) == 0:
            return np.full(self.max_length, self.padding_token, dtype=np.int64)

        nonzero_values = datapoint[nonzero_indices]

        # 2. Map to Global Vocabulary (If Dictionary Provided)
        if self.token_dictionary is not None and self.gene_names is not None:
            token_ids = []
            valid_values = []
            
            for idx, val in zip(nonzero_indices, nonzero_values):
                # Safety check for index bounds
                if idx < len(self.gene_names):
                    gene_name = self.gene_names[idx]
                    if gene_name in self.token_dictionary:
                        token_ids.append(self.token_dictionary[gene_name])
                        valid_values.append(val)
            
            final_tokens = np.array(token_ids, dtype=np.int64)
            final_values = np.array(valid_values)
        else:
            # Fallback for training from scratch
            final_tokens = nonzero_indices.astype(np.int64)
            final_values = nonzero_values

        if len(final_tokens) == 0:
             return np.full(self.max_length, self.padding_token, dtype=np.int64)

        # 3. Sort by Expression (Descending)
        # We use -final_values to sort descending
        sorted_order = np.argsort(-final_values)
        ranked_tokens = final_tokens[sorted_order]

        # 4. Truncate/Pad
        ranked_tokens = ranked_tokens[:self.max_length]
        output = np.full(self.max_length, self.padding_token, dtype=np.int64)
        output[:len(ranked_tokens)] = ranked_tokens

        return output