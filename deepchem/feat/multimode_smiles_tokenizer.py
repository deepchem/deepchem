"""
Multi-Mode SMILES Tokenizer for DeepChem

This module provides a flexible, HuggingFace-independent SMILES tokenizer
that supports multiple tokenization strategies including character-level,
atom-level, and BPE (Byte Pair Encoding) tokenization.

This addresses GitHub issue #4777:
https://github.com/deepchem/deepchem/issues/4777
"""

import re
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter, OrderedDict
import logging

logger = logging.getLogger(__name__)

# SMILES regex pattern for atom-level tokenization
# Designed by Schwaller et al. for the Molecular Transformer
ATOM_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


class MultiModeSmilesTokenizer:
    """
    A flexible SMILES tokenizer supporting multiple tokenization strategies.
    
    This tokenizer provides character-level, atom-level, and BPE tokenization
    for SMILES strings without requiring external dependencies like HuggingFace.
    
    Attributes
    ----------
    level : str
        Tokenization level: 'char', 'atom', or 'bpe'
    vocab : Dict[str, int]
        Token to index mapping
    ids_to_tokens : Dict[int, str]
        Index to token mapping
    vocab_size : int
        Number of tokens in vocabulary (for BPE mode)
    
    Examples
    --------
    >>> from deepchem.feat.multimode_smiles_tokenizer import MultiModeSmilesTokenizer
    >>> 
    >>> # Character-level tokenizer
    >>> tokenizer = MultiModeSmilesTokenizer(level='char')
    >>> tokens = tokenizer.encode("CCO")
    >>> print(tokenizer.decode(tokens))
    CCO
    >>> 
    >>> # Atom-level tokenizer (handles multi-char atoms like [Cl])
    >>> tokenizer = MultiModeSmilesTokenizer(level='atom')
    >>> tokens = tokenizer.encode("C[Cl]")
    >>> print(tokens)
    ['C', '[Cl]']
    >>> 
    >>> # BPE tokenizer with vocabulary size 1000
    >>> tokenizer = MultiModeSmilesTokenizer(level='bpe', vocab_size=1000)
    >>> tokenizer.train(["CCO", "CC(=O)O", "c1ccccc1"])  # Train on dataset
    >>> tokens = tokenizer.encode("CCO")
    
    References
    ----------
    .. [1] Philippe Schwaller et al. "Molecular Transformer: A Model for 
           Uncertainty-Calibrated Chemical Reaction Prediction"
           ACS Central Science 2019, DOI: 10.1021/acscentsci.9b00576
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    
    def __init__(
        self,
        level: str = 'atom',
        vocab_size: int = 1000,
        vocab: Optional[Dict[str, int]] = None,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
    ):
        """
        Initialize the MultiModeSmilesTokenizer.
        
        Parameters
        ----------
        level : str, default 'atom'
            Tokenization level. Options:
            - 'char': Character-level tokenization
            - 'atom': Atom-level tokenization using SMILES regex
            - 'bpe': Byte Pair Encoding (requires training)
        vocab_size : int, default 1000
            Maximum vocabulary size for BPE tokenization
        vocab : Dict[str, int], optional
            Pre-built vocabulary mapping tokens to indices.
            If None, vocabulary is built during training or on-the-fly.
        pad_token : str, default "[PAD]"
            Padding token
        unk_token : str, default "[UNK]"
            Unknown token for out-of-vocabulary items
        bos_token : str, default "[BOS]"
            Beginning of sequence token
        eos_token : str, default "[EOS]"
            End of sequence token
            
        Raises
        ------
        ValueError
            If level is not one of 'char', 'atom', or 'bpe'
        """
        if level not in ['char', 'atom', 'bpe']:
            raise ValueError(f"level must be 'char', 'atom', or 'bpe', got '{level}'")
        
        self.level = level
        self.max_vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Regex for atom-level tokenization
        self.atom_regex = re.compile(ATOM_REGEX_PATTERN)
        
        # Initialize vocabulary with special tokens
        self._special_tokens = [pad_token, unk_token, bos_token, eos_token]
        
        if vocab is not None:
            self.vocab = vocab
            self.ids_to_tokens = {v: k for k, v in vocab.items()}
        else:
            self.vocab = {tok: i for i, tok in enumerate(self._special_tokens)}
            self.ids_to_tokens = {i: tok for i, tok in enumerate(self._special_tokens)}
        
        # BPE specific attributes
        self.bpe_merges: List[Tuple[str, str]] = []
        self._trained = False
        
    @property
    def vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        """Return the padding token ID."""
        return self.vocab.get(self.pad_token, 0)
    
    @property
    def unk_token_id(self) -> int:
        """Return the unknown token ID."""
        return self.vocab.get(self.unk_token, 1)
    
    @property
    def bos_token_id(self) -> int:
        """Return the beginning of sequence token ID."""
        return self.vocab.get(self.bos_token, 2)
    
    @property
    def eos_token_id(self) -> int:
        """Return the end of sequence token ID."""
        return self.vocab.get(self.eos_token, 3)
    
    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string into a list of tokens.
        
        Parameters
        ----------
        smiles : str
            SMILES string to tokenize
            
        Returns
        -------
        List[str]
            List of tokens
            
        Examples
        --------
        >>> tokenizer = MultiModeSmilesTokenizer(level='atom')
        >>> tokenizer.tokenize("CC(=O)O")
        ['C', 'C', '(', '=', 'O', ')', 'O']
        """
        if self.level == 'char':
            return self._tokenize_char(smiles)
        elif self.level == 'atom':
            return self._tokenize_atom(smiles)
        elif self.level == 'bpe':
            return self._tokenize_bpe(smiles)
        else:
            raise ValueError(f"Unknown tokenization level: {self.level}")
    
    def _tokenize_char(self, smiles: str) -> List[str]:
        """Character-level tokenization."""
        return list(smiles)
    
    def _tokenize_atom(self, smiles: str) -> List[str]:
        """
        Atom-level tokenization using SMILES regex.
        
        Handles multi-character atoms like [Cl], [Br], stereochemistry (@, @@),
        ring numbers (%10, %11), and other SMILES syntax properly.
        """
        tokens = self.atom_regex.findall(smiles)
        return tokens
    
    def _tokenize_bpe(self, smiles: str) -> List[str]:
        """
        BPE tokenization.
        
        First applies atom-level tokenization, then applies learned BPE merges.
        """
        if not self._trained:
            logger.warning("BPE tokenizer has not been trained. Using atom-level tokenization.")
            return self._tokenize_atom(smiles)
        
        # Start with atom-level tokens
        tokens = self._tokenize_atom(smiles)
        
        # Apply BPE merges
        tokens = self._apply_bpe_merges(tokens)
        
        return tokens
    
    def _apply_bpe_merges(self, tokens: List[str]) -> List[str]:
        """Apply learned BPE merges to token list."""
        if not self.bpe_merges:
            return tokens
        
        for merge_pair in self.bpe_merges:
            tokens = self._merge_pair(tokens, merge_pair)
        
        return tokens
    
    def _merge_pair(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge all occurrences of a pair in the token list."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], "torch.Tensor"]:
        """
        Encode a SMILES string to token IDs.
        
        Parameters
        ----------
        smiles : str
            SMILES string to encode
        add_special_tokens : bool, default False
            Whether to add BOS and EOS tokens
        max_length : int, optional
            Maximum sequence length. If provided, truncates or pads to this length.
        padding : bool, default False
            Whether to pad sequences to max_length
        return_tensors : str, optional
            If 'pt', return PyTorch tensor. Otherwise return list.
            
        Returns
        -------
        Union[List[int], torch.Tensor]
            List of token IDs or PyTorch tensor
            
        Examples
        --------
        >>> tokenizer = MultiModeSmilesTokenizer(level='atom')
        >>> tokenizer.encode("CCO")
        [4, 4, 5]  # IDs depend on vocabulary
        """
        tokens = self.tokenize(smiles)
        
        # Add special tokens if requested
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert tokens to IDs, building vocab on-the-fly if needed
        ids = []
        for token in tokens:
            if token not in self.vocab:
                if self.level != 'bpe' or not self._trained:
                    # Add to vocabulary if not in BPE mode or not trained
                    self.vocab[token] = len(self.vocab)
                    self.ids_to_tokens[self.vocab[token]] = token
                else:
                    # Use UNK token if BPE is trained and token not in vocab
                    ids.append(self.unk_token_id)
                    continue
            ids.append(self.vocab[token])
        
        # Handle truncation
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        
        # Handle padding
        if padding and max_length is not None:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        
        # Convert to tensor if requested
        if return_tensors == 'pt':
            try:
                import torch
                return torch.tensor(ids)
            except ImportError:
                raise ImportError("PyTorch is required for return_tensors='pt'")
        
        return ids
    
    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to a SMILES string.
        
        Parameters
        ----------
        token_ids : Union[List[int], torch.Tensor]
            List or tensor of token IDs
        skip_special_tokens : bool, default True
            Whether to skip special tokens (PAD, UNK, BOS, EOS)
            
        Returns
        -------
        str
            Decoded SMILES string
            
        Examples
        --------
        >>> tokenizer = MultiModeSmilesTokenizer(level='atom')
        >>> ids = tokenizer.encode("CCO")
        >>> tokenizer.decode(ids)
        'CCO'
        """
        # Handle PyTorch tensors
        try:
            import torch
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
        except ImportError:
            pass
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.ids_to_tokens:
                token = self.ids_to_tokens[token_id]
                if skip_special_tokens and token in self._special_tokens:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)
        
        return ''.join(tokens)
    
    def train(
        self,
        smiles_list: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        """
        Train the tokenizer on a corpus of SMILES strings.
        
        For 'char' and 'atom' modes, this builds the vocabulary.
        For 'bpe' mode, this learns merge operations.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to train on
        vocab_size : int, optional
            Target vocabulary size (for BPE). Defaults to self.max_vocab_size
        min_frequency : int, default 2
            Minimum frequency for a token to be included in vocabulary
        verbose : bool, default False
            Whether to print training progress
            
        Examples
        --------
        >>> tokenizer = MultiModeSmilesTokenizer(level='bpe', vocab_size=500)
        >>> smiles_corpus = ["CCO", "CC(=O)O", "c1ccccc1", "CCN"]
        >>> tokenizer.train(smiles_corpus)
        >>> tokens = tokenizer.encode("CCO")
        """
        if vocab_size is not None:
            self.max_vocab_size = vocab_size
        
        # Reset vocabulary (keep special tokens)
        self.vocab = {tok: i for i, tok in enumerate(self._special_tokens)}
        self.ids_to_tokens = {i: tok for i, tok in enumerate(self._special_tokens)}
        self.bpe_merges = []
        
        if self.level in ['char', 'atom']:
            self._train_basic(smiles_list, min_frequency, verbose)
        elif self.level == 'bpe':
            self._train_bpe(smiles_list, min_frequency, verbose)
        
        self._trained = True
        
        if verbose:
            logger.info(f"Training complete. Vocabulary size: {len(self.vocab)}")
    
    def _train_basic(
        self,
        smiles_list: List[str],
        min_frequency: int,
        verbose: bool,
    ) -> None:
        """Train vocabulary for char or atom level tokenization."""
        token_counts: Counter = Counter()
        
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            token_counts.update(tokens)
        
        # Add tokens that meet minimum frequency
        for token, count in token_counts.most_common():
            if count >= min_frequency and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[token]] = token
        
        if verbose:
            logger.info(f"Built vocabulary with {len(self.vocab)} tokens")
    
    def _train_bpe(
        self,
        smiles_list: List[str],
        min_frequency: int,
        verbose: bool,
    ) -> None:
        """Train BPE tokenizer using Byte Pair Encoding algorithm."""
        # First, build initial vocabulary from atom-level tokens
        token_counts: Counter = Counter()
        
        for smiles in smiles_list:
            tokens = self._tokenize_atom(smiles)
            token_counts.update(tokens)
        
        # Add all initial tokens to vocabulary
        for token in token_counts:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[token]] = token
        
        # Tokenize all SMILES strings
        corpus = [self._tokenize_atom(smiles) for smiles in smiles_list]
        
        # Learn BPE merges
        while len(self.vocab) < self.max_vocab_size:
            # Count pair frequencies
            pair_counts: Counter = Counter()
            for tokens in corpus:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += 1
            
            if not pair_counts:
                break
            
            # Find most frequent pair
            most_common_pair = pair_counts.most_common(1)[0]
            if most_common_pair[1] < min_frequency:
                break
            
            pair = most_common_pair[0]
            self.bpe_merges.append(pair)
            
            # Add merged token to vocabulary
            merged_token = pair[0] + pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[merged_token]] = merged_token
            
            # Apply merge to corpus
            corpus = [self._merge_pair(tokens, pair) for tokens in corpus]
            
            if verbose and len(self.bpe_merges) % 100 == 0:
                logger.info(f"Learned {len(self.bpe_merges)} BPE merges, vocab size: {len(self.vocab)}")
        
        if verbose:
            logger.info(f"BPE training complete. Final vocab size: {len(self.vocab)}")
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file.
        
        Parameters
        ----------
        path : str
            Path to save the tokenizer (JSON format)
        """
        import json
        
        data = {
            'level': self.level,
            'max_vocab_size': self.max_vocab_size,
            'vocab': self.vocab,
            'bpe_merges': self.bpe_merges,
            'special_tokens': {
                'pad': self.pad_token,
                'unk': self.unk_token,
                'bos': self.bos_token,
                'eos': self.eos_token,
            },
            'trained': self._trained,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "MultiModeSmilesTokenizer":
        """
        Load a tokenizer from a file.
        
        Parameters
        ----------
        path : str
            Path to the saved tokenizer (JSON format)
            
        Returns
        -------
        MultiModeSmilesTokenizer
            Loaded tokenizer instance
        """
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(
            level=data['level'],
            vocab_size=data['max_vocab_size'],
            vocab={k: int(v) for k, v in data['vocab'].items()},
            pad_token=data['special_tokens']['pad'],
            unk_token=data['special_tokens']['unk'],
            bos_token=data['special_tokens']['bos'],
            eos_token=data['special_tokens']['eos'],
        )
        tokenizer.bpe_merges = [tuple(m) for m in data['bpe_merges']]
        tokenizer._trained = data['trained']
        
        return tokenizer
    
    def batch_encode(
        self,
        smiles_list: List[str],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[List[int]], "torch.Tensor"]:
        """
        Encode a batch of SMILES strings.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to encode
        add_special_tokens : bool, default False
            Whether to add BOS and EOS tokens
        max_length : int, optional
            Maximum sequence length
        padding : bool, default True
            Whether to pad sequences to the same length
        return_tensors : str, optional
            If 'pt', return PyTorch tensor
            
        Returns
        -------
        Union[List[List[int]], torch.Tensor]
            Batch of encoded sequences
        """
        encoded = []
        for smiles in smiles_list:
            ids = self.encode(
                smiles,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=False,
            )
            encoded.append(ids)
        
        # Determine max length for padding
        if padding:
            actual_max = max(len(seq) for seq in encoded)
            target_length = max_length if max_length is not None else actual_max
            encoded = [
                seq + [self.pad_token_id] * (target_length - len(seq))
                for seq in encoded
            ]
        
        if return_tensors == 'pt':
            try:
                import torch
                return torch.tensor(encoded)
            except ImportError:
                raise ImportError("PyTorch is required for return_tensors='pt'")
        
        return encoded
    
    def batch_decode(
        self,
        batch_ids: Union[List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token IDs.
        
        Parameters
        ----------
        batch_ids : Union[List[List[int]], torch.Tensor]
            Batch of token ID sequences
        skip_special_tokens : bool, default True
            Whether to skip special tokens
            
        Returns
        -------
        List[str]
            List of decoded SMILES strings
        """
        try:
            import torch
            if isinstance(batch_ids, torch.Tensor):
                batch_ids = batch_ids.tolist()
        except ImportError:
            pass
        
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __repr__(self) -> str:
        return (f"MultiModeSmilesTokenizer(level='{self.level}', "
                f"vocab_size={len(self.vocab)}, trained={self._trained})")
