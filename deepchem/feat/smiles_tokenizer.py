import re
import collections
import os
import json
from typing import List, Dict, Optional, Any, Union, Iterable

# Production-grade SMILES regex for atom-level parsing.
# Handles:
# 1. Bracketed atoms with isotopes, charges, hydrogens, and stereochemistry (e.g., [13C@H], [NH3+])
# 2. Multi-character atoms (Br, Cl) and single-character atoms (C, N, O, P, S, F, I, etc.)
# 3. Aromatic atoms (c, n, o, s, p, etc.)
# 4. Bonds (=, #, -, :, ., \, /)
# 5. Ring closures (1-9 and %10-%99)
# 6. Branching, stereochemistry (@, @@), and reaction arrows (>)
SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


class SmilesTokenizer:
    """
    A production-quality SMILES tokenizer for sequence models in drug discovery.

    This tokenizer supports atom-level, character-level, and BPE-level tokenization
    with built-in support for batching, vocabulary persistence, and PyTorch dataset preprocessing.

    Attributes
    ----------
    level: str
        Tokenization level: 'atom', 'char', or 'bpe'.
    pad_token: str
        Padding token.
    unk_token: str
        Unknown token.
    bos_token: str
        Beginning of sequence token.
    eos_token: str
        End of sequence token.
    vocab: Dict[str, int]
        Mapping from tokens to integer IDs.
    ids_to_tokens: Dict[int, str]
        Mapping from integer IDs back to tokens.

    Examples
    --------
    >>> from deepchem.feat.smiles_tokenizer import SmilesTokenizer
    >>> tokenizer = SmilesTokenizer(level="atom")
    >>> tokenizer.train(["CCO", "C[Cl]"])
    >>> tokens = tokenizer.encode("C[Cl]")
    >>> print(tokens)
    ['<BOS>', 'C', '[Cl]', '<EOS>']
    >>> ids = tokenizer.encode("CCO", return_ids=True)
    >>> print(tokenizer.decode(ids))
    'CCO'
    """

    def __init__(self,
                 level: str = "atom",
                 vocab_size: Optional[int] = None,
                 pad_token: str = "<PAD>",
                 unk_token: str = "<UNK>",
                 bos_token: str = "<BOS>",
                 eos_token: str = "<EOS>"):
        """
        Initialize the tokenizer.

        Parameters
        ----------
        level: str, default "atom"
            Tokenization level ('atom', 'char', 'bpe').
        vocab_size: int, optional
            Maximum vocabulary size for BPE.
        pad_token: str, default "<PAD>"
            Padding token.
        unk_token: str, default "<UNK>"
            Unknown token.
        bos_token: str, default "<BOS>"
            Beginning of sequence token.
        eos_token: str, default "<EOS>"
            End of sequence token.
        """
        self.level = level.lower()
        if self.level not in ["atom", "char", "bpe"]:
            raise ValueError("Level must be 'atom', 'char', or 'bpe'.")

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.ids_to_tokens: Dict[int, str] = {i: t for i, t in enumerate(self.special_tokens)}

        self._regex = re.compile(SMI_REGEX_PATTERN)
        self.max_vocab_size = vocab_size
        self._bpe_tokenizer = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_raw_tokens(self, smiles: str) -> List[str]:
        """Internal method to split SMILES into base tokens."""
        if self.level == "char":
            return list(smiles)
        elif self.level == "atom":
            return self._regex.findall(smiles)
        elif self.level == "bpe":
            if self._bpe_tokenizer is None:
                raise ValueError("BPE tokenizer not trained. Call train().")
            encoding = self._bpe_tokenizer.encode(smiles)
            return encoding.tokens if hasattr(encoding, 'tokens') else encoding
        return []

    def encode(self,
               smiles: str,
               add_special_tokens: bool = True,
               return_ids: bool = False) -> List[Any]:
        """
        Tokenize a single SMILES string.

        Parameters
        ----------
        smiles: str
            Input SMILES.
        add_special_tokens: bool, default True
            Whether to wrap with <BOS> and <EOS>.
        return_ids: bool, default False
            Return integer IDs instead of strings.
        """
        tokens = self._get_raw_tokens(smiles)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]

        if return_ids:
            return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]
        return tokens

    def batch_encode(self,
                     smiles_list: Iterable[str],
                     add_special_tokens: bool = True,
                     return_ids: bool = False) -> List[List[Any]]:
        """Efficiently tokenize a batch of SMILES strings."""
        return [self.encode(s, add_special_tokens, return_ids) for s in smiles_list]

    def decode(self, tokens: List[Union[str, int]]) -> str:
        """Reconstruct SMILES from tokens, stripping special tokens."""
        if not tokens:
            return ""

        if isinstance(tokens[0], int):
            tokens = [self.ids_to_tokens.get(i, self.unk_token) for i in tokens]

        # Remove special tokens for reconstruction
        clean_tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(clean_tokens)

    def train(self, smiles_list: Iterable[str]):
        """Train the vocabulary on a corpus of SMILES."""
        if self.level in ["atom", "char"]:
            unique_tokens = set()
            for s in smiles_list:
                unique_tokens.update(self._get_raw_tokens(s))

            for t in sorted(list(unique_tokens)):
                if t not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[t] = idx
                    self.ids_to_tokens[idx] = t
        elif self.level == "bpe":
            try:
                from tokenizers import Tokenizer
                from tokenizers.models import BPE
                from tokenizers.trainers import BpeTrainer
                from tokenizers.pre_tokenizers import Split

                tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
                tokenizer.pre_tokenizer = Split(re_pattern=SMI_REGEX_PATTERN,
                                                behavior="isolated")
                trainer = BpeTrainer(vocab_size=self.max_vocab_size or 5000,
                                     special_tokens=self.special_tokens)
                tokenizer.train_from_iterator(smiles_list, trainer=trainer)
                self._bpe_tokenizer = tokenizer
                self.vocab = tokenizer.get_vocab()
                self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
            except ImportError:
                print("Tokenizers library not found. Falling back to atom-level.")
                self.level = "atom"
                self.train(smiles_list)

    def save_vocab(self, path: str):
        """Persist the vocabulary to a JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        """Load vocabulary from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            self.ids_to_tokens = {int(v): k for k, v in self.vocab.items()}
        # Recover special tokens from loaded vocab
        inv_vocab = {v: k for k, v in self.vocab.items()}
        # We assume indices 0-3 were set during initialization but let's be robust
        def find_token(t): return inv_vocab.get(self.vocab.get(t, -1), t)
        self.pad_token = find_token(self.pad_token)
        # Verify mandatory special tokens exist
        for t in self.special_tokens:
            if t not in self.vocab:
                raise ValueError(f"Required special token {t} missing from loaded vocab.")

    def tokenize_dataset(self,
                         smiles_list: Iterable[str],
                         max_length: int) -> Any:
        """
        Preprocess a dataset into a padded PyTorch tensor of token IDs.

        Returns
        -------
        torch.Tensor
            A tensor of shape (len(smiles_list), max_length).
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for tokenize_dataset.")

        batch_ids = []
        pad_id = self.vocab[self.pad_token]

        for s in smiles_list:
            ids = self.encode(s, add_special_tokens=True, return_ids=True)
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [pad_id] * (max_length - len(ids))
            batch_ids.append(ids)

        return torch.tensor(batch_ids)


# For backward compatibility
def load_vocab(path: str) -> Dict[str, int]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class BasicSmilesTokenizer:
    """Simplified tokenizer for basic parsing needs."""
    def __init__(self):
        self._regex = re.compile(SMI_REGEX_PATTERN)

    def tokenize(self, smiles: str) -> List[str]:
        return self._regex.findall(smiles)
