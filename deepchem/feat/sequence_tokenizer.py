import numpy as np
from typing import List, Dict, Optional
from deepchem.feat import Featurizer


class DNACharTokenizer(Featurizer):
    """
    Character-level DNA sequence tokenizer.

    Converts DNA sequences into integer token IDs with padding and
    attention masks suitable for transformer-based models.

    Notes
    -----
    This tokenizer does not perform model-specific preprocessing.
    It is intended as a lightweight, reusable sequence tokenizer
    for DNA foundation models.

    Example
    -------
    >>> tokenizer = DNACharTokenizer(max_length=8)
    >>> features = tokenizer.featurize(["ACGT", "A"])
    >>> features["input_ids"].shape
    (2, 8)
    """
    def __init__(self, max_length: Optional[int] = None, pad_token: str = "<pad>", mask_token: str = "<mask>"):
        self.vocab = {
            pad_token: 0,
            mask_token: 1,
            "A": 2,
            "C": 3,
            "G": 4,
            "T": 5,
            "N": 6,
        }
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.pad_id = self.vocab[pad_token]
        self.mask_token_id = self.vocab[mask_token]
        self.max_length = max_length

    def _tokenize(self, seq: str) -> List[int]:
        return [self.vocab.get(ch.upper(), self.vocab["N"]) for ch in seq]

    def featurize(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        tokenized = [self._tokenize(s) for s in sequences]

        max_len = self.max_length or max(len(t) for t in tokenized)
        input_ids = np.full((len(tokenized), max_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(tokenized), max_len), dtype=np.int64)

        for i, tokens in enumerate(tokenized):
            trunc = tokens[:max_len]
            input_ids[i, :len(trunc)] = trunc
            attention_mask[i, :len(trunc)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class DNAKmerTokenizer(Featurizer):
    """
    k-mer based DNA sequence tokenizer.

    Splits DNA sequences into overlapping k-mers and converts them
    into integer token IDs with padding and attention masks.

    Examples
    --------
    >>> tokenizer = DNAKmerTokenizer(k=2, max_length=4)
    >>> features = tokenizer.featurize(["ACGT"])
    >>> features["input_ids"].shape
    (1, 4)

    Sequence: ACGT, k=2 -> [AC, CG, GT]
    """
    def __init__(self, k: int = 3, max_length: Optional[int] = None, pad_token: str = "<pad>", mask_token: str = "<mask>"):
        self.k = k
        self.max_length = max_length
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.vocab = {pad_token: 0, mask_token: 1}
        self.pad_id = self.vocab[pad_token]
        self.mask_token_id = self.vocab[mask_token]
        self._frozen = False

    def _build_vocab(self, sequences: List[str]):
        idx = 1
        for seq in sequences:
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k].upper()
                if kmer not in self.vocab:
                    self.vocab[kmer] = idx
                    idx += 1
        self._frozen = True

    def _tokenize(self, seq: str) -> List[int]:
        tokens = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i + self.k].upper()
            tokens.append(self.vocab.get(kmer, 0))
        return tokens

    def featurize(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        if not self._frozen:
            self._build_vocab(sequences)

        tokenized = [self._tokenize(s) for s in sequences]
        max_len = self.max_length or max(len(t) for t in tokenized)

        input_ids = np.zeros((len(tokenized), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(tokenized), max_len), dtype=np.int64)

        for i, tokens in enumerate(tokenized):
            trunc = tokens[:max_len]
            input_ids[i, :len(trunc)] = trunc
            attention_mask[i, :len(trunc)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        } 
