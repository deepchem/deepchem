"""IUPAC name featurizer for molecular property prediction and translation tasks."""

import logging
from typing import Iterable, List, Optional
import numpy as np

from deepchem.feat.base_classes import Featurizer, MolecularFeaturizer

logger = logging.getLogger(__name__)

# Character vocabulary covering all common characters in PubChem IUPAC names
IUPAC_CHARSET = [
    '<pad>', '<unk>', '<sos>', '<eos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '(', ')', ',', '[', ']',
    "'", ' '
]

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'


class IUPACFeaturizer(MolecularFeaturizer):
    """Featurizer for IUPAC chemical names.

    Converts IUPAC names into integer index sequences suitable for
    sequence-to-sequence models. This featurizer is the IUPAC-side
    counterpart to :class:`SmilesToSeq` and is intended for use in
    bidirectional SMILES<->IUPAC translation tasks.

    Parameters
    ----------
    max_len : int, optional (default 200)
        Maximum allowed length of the IUPAC name (excluding special tokens).
    charset : list of str, optional
        Character vocabulary. Defaults to IUPAC_CHARSET which covers all
        characters found in PubChem IUPAC names.

    References
    ----------
    .. [1] Kim, S. et al. "PubChem 2023 update." Nucleic Acids Research,
       2023. https://doi.org/10.1093/nar/gkac956
    """

    def __init__(self, max_len: int = 200, charset: Optional[List[str]] = None):
        """Initialize IUPACFeaturizer"""
        self.max_len = max_len
        self.charset = charset if charset is not None else IUPAC_CHARSET
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.charset)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.charset)}
        self.pad_idx = self.char_to_idx[PAD_TOKEN]
        self.unk_idx = self.char_to_idx[UNK_TOKEN]
        self.sos_idx = self.char_to_idx[SOS_TOKEN]
        self.eos_idx = self.char_to_idx[EOS_TOKEN]

    def featurize(self, datapoints: Iterable[str], log_every_n: int = 1000, **kwargs) -> np.ndarray:
        """Calculate features for a sequence of IUPAC name strings"""
        return Featurizer.featurize(self, datapoints, log_every_n, **kwargs)

    def _featurize(self, datapoint: str, **kwargs) -> np.ndarray:
        """Featurize a single IUPAC name string"""
        if not isinstance(datapoint, str):
            raise ValueError(
                f"IUPACFeaturizer expects a string, got {type(datapoint)}.")

        if len(datapoint) > self.max_len:
            logger.warning(
                f"IUPAC name of length {len(datapoint)} exceeds max_len "
                f"({self.max_len}). Truncating.")
            datapoint = datapoint[:self.max_len]

        # Encode each character, falling back to <unk> for out-of-vocab chars
        encoded = [self.char_to_idx.get(ch, self.unk_idx) for ch in datapoint]

        # Right-pad to max_len
        pad_length = self.max_len - len(encoded)
        encoded = encoded + [self.pad_idx] * pad_length

        # Wrap with <sos> at the start and <eos> at the end
        encoded = [self.sos_idx] + encoded + [self.eos_idx]

        return np.array(encoded, dtype=np.int32)

    def decode(self, indices: np.ndarray) -> str:
        """Decode an index array back to an IUPAC name string."""
        special = {self.pad_idx, self.sos_idx, self.eos_idx}
        chars = [
            self.idx_to_char.get(idx, UNK_TOKEN)
            for idx in indices
            if idx not in special
        ]
        return ''.join(chars)

    @property
    def vocab_size(self) -> int:
        """Returns the size of the character vocabulary."""
        return len(self.charset)
