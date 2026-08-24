"""
Unified SMILES Tokenizer Utility for Sequence Models.

This module provides a flexible, dependency-light ``SmilesTokenizerV2`` class
that supports three tokenization strategies:

* ``'char'``  – split every character individually
* ``'atom'``  – regex-based splitting that respects multi-char atoms and
  SMILES syntax (Br, Cl, [NH3+], @@, ring closures, …)
* ``'bpe'``   – Byte-Pair Encoding trained on a SMILES corpus via the
  ``tokenizers`` library (optional dependency)

The class is **transformer-library-independent** for the ``char`` and
``atom`` strategies, mirroring the philosophy of the existing
:class:`BasicSmilesTokenizer` in DeepChem while adding:

* a proper vocabulary with integer ``encode`` / ``decode``
* optional PyTorch tensor output
* padding support
* BPE training on arbitrary SMILES corpora

References
----------
.. [1] Philippe Schwaller et al. ACS Central Science 2019 5(9):1572-1583
       DOI: 10.1021/acscentsci.9b00576

Examples
--------
Character-level tokenizer::

    >>> from deepchem.feat.smiles_tokenizer_v2 import SmilesTokenizerV2
    >>> tokenizer = SmilesTokenizerV2(level='char')
    >>> tokenizer.tokenize("CCO")
    ['C', 'C', 'O']
    >>> ids = tokenizer.encode("CCO")
    >>> tokenizer.decode(ids)
    'CCO'

Atom-level tokenizer (handles multi-char atoms)::

    >>> tokenizer = SmilesTokenizerV2(level='atom')
    >>> tokenizer.tokenize("C[Cl]")
    ['C', '[Cl]']

BPE tokenizer::

    >>> tokenizer = SmilesTokenizerV2(level='bpe', vocab_size=100)
    >>> tokenizer.train(["CCO", "CC(=O)O", "c1ccccc1"])
    >>> ids = tokenizer.encode("CCO")
    >>> tokenizer.decode(ids)
    'CCO'
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# SMILES regex developed by Schwaller et al. (same as BasicSmilesTokenizer)
# ---------------------------------------------------------------------------
SMI_REGEX_PATTERN = (
    r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"""
    r"""|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
)

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class SmilesTokenizerV2:
    """Unified SMILES tokenizer supporting ``char``, ``atom``, and ``bpe`` strategies.

    This tokenizer is designed to serve as a reusable building block for
    sequence-based models operating on SMILES strings (e.g. SMILES↔IUPAC
    translation, MolBERT-style architectures).

    Parameters
    ----------
    level : str
        Tokenization strategy.  One of ``'char'``, ``'atom'``, ``'bpe'``.
        Default is ``'atom'``.
    vocab_size : int, optional
        Target vocabulary size when ``level='bpe'``.  Ignored otherwise.
        Default is ``1000``.
    add_sos_eos : bool, optional
        Whether to prepend ``<sos>`` and append ``<eos>`` during encoding.
        Default is ``False``.
    pad_token : str, optional
        Padding token string.  Default is ``'<pad>'``.
    unk_token : str, optional
        Unknown-token string.  Default is ``'<unk>'``.

    Attributes
    ----------
    vocab : Dict[str, int]
        Mapping from token string to integer index.
    inverse_vocab : Dict[int, str]
        Reverse mapping from integer index to token string.

    Notes
    -----
    The ``'bpe'`` strategy requires the ``tokenizers`` library
    (``pip install tokenizers``).  The ``'char'`` and ``'atom'`` strategies
    have **no extra dependencies** beyond the Python standard library.
    """

    def __init__(
        self,
        level: str = "atom",
        vocab_size: int = 1000,
        add_sos_eos: bool = False,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
    ) -> None:
        if level not in ("char", "atom", "bpe"):
            raise ValueError(
                f"level must be 'char', 'atom', or 'bpe', got '{level}'"
            )

        self.level = level
        self.vocab_size = vocab_size
        self.add_sos_eos = add_sos_eos
        self.pad_token = pad_token
        self.unk_token = unk_token

        self._regex = re.compile(SMI_REGEX_PATTERN)

        # Vocabulary – populated by build_vocab() or train()
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

        # BPE tokenizer (lazy-imported, only used when level='bpe')
        self._bpe_tokenizer = None

        # Build a default vocabulary for char/atom from a canonical SMILES
        # character set so the tokenizer works out-of-the-box without
        # requiring a call to build_vocab() for simple use-cases.
        if level in ("char", "atom"):
            self._build_default_vocab()

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    def _build_default_vocab(self) -> None:
        """Seed the vocabulary with common SMILES tokens."""
        # Collected from common organic / inorganic SMILES
        default_tokens = list("BCFINOPSbcnosp") + [
            "Cl", "Br",
            "[NH]", "[NH2]", "[NH3+]", "[NH4+]",
            "[OH]", "[OH2]", "[OH-]",
            "[CH]", "[CH2]", "[CH3]",
            "[nH]",
            "[N+]", "[N-]", "[O+]", "[O-]", "[S+]", "[S-]",
            "[Na+]", "[K+]", "[Ca+2]", "[Mg+2]", "[Cl-]", "[Br-]",
            "(", ")", "[", "]",
            "=", "#", "-", "+", "\\", "/", ":", "~", "@", "@@",
            ".", ">",
            "%10", "%11", "%12",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        ]
        self.build_vocab(default_tokens)

    def build_vocab(self, tokens: List[str]) -> None:
        """Build the vocabulary from a list of token strings.

        Special tokens are always prepended at indices 0-3.

        Parameters
        ----------
        tokens : List[str]
            Ordered list of token strings (duplicates are deduplicated while
            preserving first-seen order).
        """
        seen = dict.fromkeys(SPECIAL_TOKENS)  # preserves insertion order
        for tok in tokens:
            if tok not in seen:
                seen[tok] = None
        self.vocab = {tok: idx for idx, tok in enumerate(seen)}
        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def build_vocab_from_smiles(self, smiles_list: List[str]) -> None:
        """Build vocabulary by tokenising a list of SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            Collection of SMILES strings used to derive the token set.
        """
        token_counts: Counter = Counter()
        for smi in smiles_list:
            token_counts.update(self.tokenize(smi))

        # Sort by frequency descending so common tokens get low IDs
        sorted_tokens = [tok for tok, _ in token_counts.most_common()]
        self.build_vocab(sorted_tokens)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, smiles: str) -> List[str]:
        """Convert a SMILES string to a list of token strings.

        Parameters
        ----------
        smiles : str
            A valid SMILES string.

        Returns
        -------
        List[str]
            Ordered list of token strings.

        Examples
        --------
        >>> t = SmilesTokenizerV2(level='atom')
        >>> t.tokenize("C[Cl]")
        ['C', '[Cl]']
        """
        if self.level == "char":
            return list(smiles)
        elif self.level == "atom":
            tokens = self._regex.findall(smiles)
            if not tokens:
                # Fall back to character-level for unrecognised patterns
                return list(smiles)
            return tokens
        else:  # bpe
            if self._bpe_tokenizer is None:
                raise RuntimeError(
                    "BPE tokenizer has not been trained yet. "
                    "Call .train(smiles_list) first."
                )
            return self._bpe_tokenizer.encode(smiles).tokens

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(
        self,
        smiles: str,
        pad_length: Optional[int] = None,
        return_tensor: bool = False,
    ) -> Union[List[int], "torch.Tensor"]:  # noqa: F821
        """Tokenise and convert a SMILES string to integer token IDs.

        Parameters
        ----------
        smiles : str
            Input SMILES string.
        pad_length : int, optional
            If provided, the output is padded (or truncated) to this length
            using ``self.pad_token``.
        return_tensor : bool, optional
            If ``True``, return a ``torch.Tensor`` instead of a list.
            Requires PyTorch to be installed.

        Returns
        -------
        List[int] or torch.Tensor
            Integer token IDs.

        Examples
        --------
        >>> t = SmilesTokenizerV2(level='atom')
        >>> ids = t.encode("CCO")
        >>> isinstance(ids, list)
        True
        """
        tokens = self.tokenize(smiles)

        if self.add_sos_eos:
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]

        unk_id = self.vocab.get(UNK_TOKEN, 1)
        ids = [self.vocab.get(tok, unk_id) for tok in tokens]

        if pad_length is not None:
            pad_id = self.vocab.get(PAD_TOKEN, 0)
            if len(ids) < pad_length:
                ids = ids + [pad_id] * (pad_length - len(ids))
            else:
                ids = ids[:pad_length]

        if return_tensor:
            try:
                import torch  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "PyTorch is required for return_tensor=True. "
                    "Install it with: pip install torch"
                ) from exc
            return torch.tensor(ids, dtype=torch.long)

        return ids

    def encode_batch(
        self,
        smiles_list: List[str],
        pad: bool = True,
        return_tensor: bool = False,
    ) -> Union[List[List[int]], "torch.Tensor"]:  # noqa: F821
        """Encode a batch of SMILES strings, optionally padding to equal length.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings.
        pad : bool, optional
            If ``True`` (default), pad all sequences to the length of the
            longest one in the batch.
        return_tensor : bool, optional
            Return a 2-D ``torch.Tensor`` of shape ``(batch, seq_len)``.

        Returns
        -------
        List[List[int]] or torch.Tensor
        """
        encoded = [self.encode(smi) for smi in smiles_list]
        if pad:
            max_len = max(len(e) for e in encoded)
            pad_id = self.vocab.get(PAD_TOKEN, 0)
            encoded = [
                e + [pad_id] * (max_len - len(e)) for e in encoded
            ]
        if return_tensor:
            try:
                import torch  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "PyTorch is required for return_tensor=True."
                ) from exc
            return torch.tensor(encoded, dtype=torch.long)
        return encoded

    def decode(
        self, token_ids: Union[List[int], "torch.Tensor"]  # noqa: F821
    ) -> str:
        """Convert integer token IDs back to a SMILES string.

        Special tokens (``<pad>``, ``<sos>``, ``<eos>``, ``<unk>``) are
        removed from the output.

        Parameters
        ----------
        token_ids : List[int] or torch.Tensor
            Sequence of integer token IDs.

        Returns
        -------
        str
            Reconstructed SMILES string.

        Examples
        --------
        >>> t = SmilesTokenizerV2(level='atom')
        >>> ids = t.encode("CC(=O)O")
        >>> t.decode(ids)
        'CC(=O)O'
        """
        # Handle tensors gracefully
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        skip = {self.vocab.get(st) for st in SPECIAL_TOKENS if st in self.vocab}
        tokens = [
            self.inverse_vocab[tid]
            for tid in token_ids
            if tid in self.inverse_vocab and tid not in skip
        ]
        return "".join(tokens)

    # ------------------------------------------------------------------
    # BPE training
    # ------------------------------------------------------------------

    def train(self, smiles_list: List[str]) -> None:
        """Train a BPE vocabulary on a corpus of SMILES strings.

        This method is only applicable when ``level='bpe'``.  It uses the
        ``tokenizers`` library under the hood.

        Parameters
        ----------
        smiles_list : List[str]
            SMILES strings to train on.

        Raises
        ------
        ValueError
            If ``level`` is not ``'bpe'``.
        ImportError
            If the ``tokenizers`` library is not installed.
        """
        if self.level != "bpe":
            raise ValueError("train() is only applicable when level='bpe'.")

        try:
            from tokenizers import Regex, Tokenizer  # type: ignore[import]
            from tokenizers.models import BPE  # type: ignore[import]
            from tokenizers.pre_tokenizers import Split  # type: ignore[import]
            from tokenizers.trainers import BpeTrainer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'tokenizers' library is required for BPE training. "
                "Install with: pip install tokenizers"
            ) from exc

        # Atom-level pre-tokenization keeps chemical meaning intact
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Split(  # type: ignore[assignment]
            pattern=Regex(SMI_REGEX_PATTERN),
            behavior="isolated",
        )

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tokenizer.train_from_iterator(smiles_list, trainer=trainer)
        self._bpe_tokenizer = tokenizer

        # Sync self.vocab / self.inverse_vocab
        vocab_dict = tokenizer.get_vocab()
        self.vocab = vocab_dict
        self.inverse_vocab = {v: k for k, v in vocab_dict.items()}

    # ------------------------------------------------------------------
    # Vocabulary properties
    # ------------------------------------------------------------------

    @property
    def vocab_len(self) -> int:
        """Number of tokens in the vocabulary (including special tokens)."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        """Integer ID of the padding token."""
        return self.vocab.get(PAD_TOKEN, 0)

    @property
    def unk_id(self) -> int:
        """Integer ID of the unknown token."""
        return self.vocab.get(UNK_TOKEN, 1)

    @property
    def sos_id(self) -> int:
        """Integer ID of the start-of-sequence token."""
        return self.vocab.get(SOS_TOKEN, 2)

    @property
    def eos_id(self) -> int:
        """Integer ID of the end-of-sequence token."""
        return self.vocab.get(EOS_TOKEN, 3)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_vocab(self, path: str) -> None:
        """Save the vocabulary to a plain-text file (one token per line).

        Parameters
        ----------
        path : str
            Destination file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            for tok in self.vocab:
                fh.write(tok + "\n")

    @classmethod
    def load_vocab(
        cls, path: str, level: str = "atom", **kwargs
    ) -> "SmilesTokenizerV2":
        """Load a tokenizer from a saved vocabulary file.

        Parameters
        ----------
        path : str
            Path to a vocabulary file (one token per line).
        level : str
            Tokenization level for the new tokenizer.
        **kwargs
            Additional keyword arguments passed to ``__init__``.

        Returns
        -------
        SmilesTokenizerV2
        """
        with open(path, "r", encoding="utf-8") as fh:
            tokens = [line.rstrip("\n") for line in fh if line.strip()]

        tokenizer = cls(level=level, **kwargs)
        tokenizer.build_vocab(tokens)
        return tokenizer

    def __repr__(self) -> str:
        return (
            f"SmilesTokenizerV2(level='{self.level}', "
            f"vocab_size={self.vocab_len}, "
            f"add_sos_eos={self.add_sos_eos})"
        )