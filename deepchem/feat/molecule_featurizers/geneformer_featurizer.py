"""
Geneformer Featurizer for single-cell gene expression data.
"""
import pickle
import os
from typing import Dict, List, Optional, Union

import numpy as np

from deepchem.feat.base_classes import Featurizer

class GeneformerFeaturizer(Featurizer):
    """Rank-Value Encoding featurizer for Geneformer (V1 & V2).

    This featurizer transforms raw gene expression count vectors into
    token ID sequences using the specific Rank-Value Encoding required by
    the Geneformer model.

    The encoding process prioritizes genes that distinguish cell state by
    normalizing expression counts by the global median expression of that
    gene across the training corpus.

    The encoding algorithm:
    1. Normalize raw counts by total read depth in the cell (n_counts).
    2. Normalize relative expression by the global gene median.
    3. Sort genes by this normalized 'rank value' (descending).
    4. Map genes to Token IDs.
    5. (V2 Only) Insert <cls> and <eos> special tokens.
    6. Truncate or pad sequence to fixed length.

    Parameters
    ----------
    gene_median_file : str
        Path to the pickle file containing the gene median dictionary
        (Ensembl ID -> Median Value). Required to calculate rank values.
        Can be downloaded from Hugging Face (ctheodoris/Geneformer).
    token_dictionary_file : str
        Path to the pickle file containing the gene-to-token dictionary
        (Ensembl ID -> Integer).
        Can be downloaded from Hugging Face (ctheodoris/Geneformer).
    gene_names : List[str]
        A list of gene names (Ensembl IDs) corresponding to the columns
        of your input array.
    max_length : int, optional
        Maximum sequence length.
        - If None and model_version='V1', defaults to 2048.
        - If None and model_version='V2', defaults to 4096.
    model_version : str, default 'V2'
        The model version to target ('V1' or 'V2').
        - 'V1': 2048 token length, no special tokens.
        - 'V2': 4096 token length, adds <cls> and <eos> tokens.
    padding_token_id : int, default 0
        Token ID used for padding.

    Examples
    --------
    >>> # Mock setup (In practice, download .pkl files from Hugging Face)
    >>> import numpy as np
    >>> gene_names = ["ENSG001", "ENSG002", "ENSG003"]
    >>> # featurizer = GeneformerFeaturizer(
    >>> #     gene_median_file="gene_median_dictionary.pkl",
    >>> #     token_dictionary_file="token_dictionary.pkl",
    >>> #     gene_names=gene_names,
    >>> #     model_version="V2"
    >>> # )
    >>> # feats = featurizer.featurize(raw_counts_array)

    Notes
    -----
    - Input data must be **Raw Counts** (integers), not log-normalized.
    - Input gene names must be **Ensembl IDs**.
    """

    def __init__(self,
                 gene_median_file: str,
                 token_dictionary_file: str,
                 gene_names: List[str],
                 max_length: Optional[int] = None,
                 model_version: str = "V2",
                 padding_token_id: int = 0) -> None:
        
        self.gene_names = gene_names
        self.model_version = model_version.upper()
        self.padding_token_id = padding_token_id

        if self.model_version not in ["V1", "V2"]:
            raise ValueError(f"model_version must be 'V1' or 'V2', got {model_version}")

        if max_length is None:
            self.max_length = 4096 if self.model_version == "V2" else 2048
        else:
            self.max_length = max_length

        with open(token_dictionary_file, "rb") as f:
            self.token_dictionary: Dict[str, int] = pickle.load(f)

        with open(gene_median_file, "rb") as f:
            self.gene_median_dict: Dict[str, float] = pickle.load(f)

        self.cls_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None

        if self.model_version == "V2":
            self.cls_token_id = self.token_dictionary.get("<cls>")
            self.eos_token_id = self.token_dictionary.get("<eos>")
            
            # Critical Check for V2 integrity
            if self.cls_token_id is None or self.eos_token_id is None:
                raise ValueError(
                    "model_version='V2' selected, but <cls> or <eos> tokens "
                    "missing from token_dictionary_file. Are you using a V1 dictionary?"
                )

    def _featurize(self, datapoint: np.ndarray) -> np.ndarray:
        """Apply Rank-Value Encoding to a single gene expression vector."""
        
        # Ensure input is a 1D numpy array
        if not isinstance(datapoint, np.ndarray):
            datapoint = np.array(datapoint)
        if datapoint.ndim > 1:
            datapoint = datapoint.flatten()

        total_counts = np.sum(datapoint)
        
        # Edge case: Empty cell
        if total_counts == 0:
            return np.full(self.max_length, self.padding_token_id, dtype=np.int64)

        nonzero_indices = np.flatnonzero(datapoint)
        nonzero_values = datapoint[nonzero_indices]

        token_ids: List[int] = []
        rank_values: List[float] = []

        target_sum = 10000 # the default value used in tokenizer.py of Geneformer

        for idx, raw_count in zip(nonzero_indices, nonzero_values):
        
            if idx < len(self.gene_names):
                gene_name = self.gene_names[idx]

                if (gene_name in self.token_dictionary and 
                    gene_name in self.gene_median_dict):
                    
                    depth_norm = raw_count / total_counts * target_sum
                    
                    median = self.gene_median_dict[gene_name]
                    
                    if median > 0:
                        score = depth_norm / median
                        
                        token_ids.append(self.token_dictionary[gene_name])
                        rank_values.append(score)

        # Edge case: No valid genes found in dictionary
        if not token_ids:
            return np.full(self.max_length, self.padding_token_id, dtype=np.int64)

        sorted_indices = np.argsort(-np.array(rank_values))
        ranked_tokens = np.array(token_ids, dtype=np.int64)[sorted_indices]

        output = np.full(self.max_length, self.padding_token_id, dtype=np.int64)

        if self.model_version == "V2":
            # For V2: <CLS> + Genes... + <EOS>
            # We have 2 special tokens, so we can fit (max_length - 2) genes
            max_genes = self.max_length - 2
            
            truncated_genes = ranked_tokens[:max_genes]
            
            # Construct the sandwich
            sequence = np.concatenate([
                [self.cls_token_id], 
                truncated_genes, 
                [self.eos_token_id]
            ])
        else:
            # For V1: Just Genes...
            truncated_genes = ranked_tokens[:self.max_length]
            sequence = truncated_genes

        # Place into output array
        output[:len(sequence)] = sequence
        
        return output