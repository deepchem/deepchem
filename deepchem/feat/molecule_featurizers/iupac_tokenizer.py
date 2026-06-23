"""Word-level tokenizer for IUPAC chemical nomenclature."""
import re
from collections import OrderedDict
from typing import Dict, List

# All word-level IUPAC tokens sorted longest-first so that longer matches
# take precedence over shorter overlapping ones in the regex alternation.
_IUPAC_VOCAB: List[str] = sorted(
    [
        # Multiplying prefixes
        'tetrakis', 'tris', 'bis',
        'deca', 'nona', 'octa', 'hepta', 'hexa', 'penta', 'tetra', 'tri', 'di',
        # Positional/iso prefixes
        'tert', 'sec', 'neo', 'iso',
        # Cycloalkyl substituents and parent rings
        'cyclohexane', 'cyclopentane', 'cyclobutane', 'cyclopropane',
        'cyclohexyl', 'cyclopentyl', 'cyclobutyl', 'cyclopropyl', 'cyclo',
        # Named ring systems
        'naphthalene', 'benzene', 'toluene', 'phenyl', 'benzyl',
        # Functional-group prefixes
        'methoxy', 'ethoxy', 'propoxy', 'butoxy',
        'hydroxy', 'amino', 'chloro', 'bromo', 'fluoro', 'iodo', 'nitro',
        'oxo', 'thio',
        # Alkyl substituents C1–C10
        'decyl', 'nonyl', 'octyl', 'heptyl', 'hexyl', 'pentyl',
        'butyl', 'propyl', 'ethyl', 'methyl',
        # Full parent-chain alkane names C1–C10
        'decane', 'nonane', 'octane', 'heptane', 'hexane', 'pentane',
        'butane', 'propane', 'ethane', 'methane',
        # Connecting chain forms (used in anol/anone/anoic contexts)
        'decan', 'nonan', 'octan', 'heptan', 'hexan', 'pentan',
        'butan', 'propan', 'ethan', 'methan',
        # Bare chain roots C1–C10
        'dec', 'non', 'oct', 'hept', 'hex', 'pent', 'but', 'prop', 'eth', 'meth',
        # Compound and simple suffixes (longest first within this tier)
        'amine', 'amide', 'anoic', 'anone', 'anol',
        'ene', 'yne', 'oyl', 'oxy', 'oic', 'ole', 'ate', 'ane', 'one', 'ol', 'al', 'yl',
    ],
    key=len,
    reverse=True,
)

# Single compiled pattern — priority order (first match wins):
#   1. Stereodescriptors in parentheses: (R) (S) (E) (Z) (+) (-)
#   2. Multi-word suffix: "oic acid"
#   3. All IUPAC word tokens (longest first)
#   4. Locants: digit sequences
#   5. Punctuation separators
#   6. Fallback: any single character
_PATTERN = re.compile(
    r'\((?:R|S|E|Z|\+|-)\)'
    r'|oic acid'
    r'|' + '|'.join(re.escape(t) for t in _IUPAC_VOCAB)
    + r'|\d+|[-,()\[\]\']|.'
)

_UNK = '[UNK]'
_UNK_ID = 0


class IUPACTokenizer:
    """Word-level tokenizer for IUPAC chemical nomenclature.

    Splits IUPAC names into meaningful units — multiplying prefixes,
    substituent prefixes, chain roots, functional-group suffixes,
    locants, and stereodescriptors — using a single ordered regex with
    no external dependencies.

    Vocabulary is empty on construction; call :meth:`build_vocab` to
    populate it before using :meth:`encode` or :meth:`decode`.
    ``[UNK]`` is always reserved at index 0.

    Examples
    --------
    >>> tok = IUPACTokenizer()
    >>> tok.tokenize("2-methylpropan-1-ol")
    ['2', '-', 'methyl', 'propan', '-', '1', '-', 'ol']
    >>> tok.build_vocab(["methane", "ethanol"])
    >>> tok.decode(tok.encode("methane"))
    'methane'
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.ids_to_tokens: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Core tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, name: str) -> List[str]:
        """Split an IUPAC name into a list of string tokens.

        Parameters
        ----------
        name : str
            IUPAC chemical name.

        Returns
        -------
        List[str]
            Ordered list of tokens.
        """
        return _PATTERN.findall(name)

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def build_vocab(self, names: List[str]) -> None:
        """Build a vocabulary from a corpus of IUPAC names.

        Assigns integer IDs to every unique token observed.
        ``[UNK]`` is always reserved at index 0.

        Parameters
        ----------
        names : List[str]
            Corpus of IUPAC chemical names.
        """
        seen: OrderedDict = OrderedDict()
        for name in names:
            for tok in self.tokenize(name):
                seen[tok] = None
        self.vocab = {_UNK: _UNK_ID}
        for i, tok in enumerate(seen, start=1):
            self.vocab[tok] = i
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Map tokens to integer IDs; unknown tokens map to 0 (``[UNK]``)."""
        return [self.vocab.get(t, _UNK_ID) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Map integer IDs back to tokens; unknown IDs map to ``[UNK]``."""
        return [self.ids_to_tokens.get(i, _UNK) for i in ids]

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, name: str) -> List[int]:
        """Tokenize an IUPAC name and map each token to an integer ID.

        Parameters
        ----------
        name : str
            IUPAC chemical name.

        Returns
        -------
        List[int]
            Integer ID sequence; out-of-vocabulary tokens map to 0.
        """
        return self.convert_tokens_to_ids(self.tokenize(name))

    def decode(self, ids: List[int]) -> str:
        """Reconstruct an IUPAC name from a sequence of integer IDs.

        Parameters
        ----------
        ids : List[int]
            Sequence of integer token IDs.

        Returns
        -------
        str
            Reconstructed IUPAC name via token concatenation.
        """
        return ''.join(self.convert_ids_to_tokens(ids))
