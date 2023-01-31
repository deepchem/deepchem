# Requriments - transformers, tokenizers
# Right now, the Smiles Tokenizer uses an exiesting vocab file from rxnfp that is fairly comprehensive and from the USPTO dataset.
# The vocab may be expanded in the near future

import collections
import os
import re
import pkg_resources
from typing import List, Optional
from transformers import BertTokenizer
from logging import getLogger

logger = getLogger(__name__)
"""
SMI_REGEX_PATTERN: str
SMILES regex pattern for tokenization. Designed by Schwaller et. al.

References
----------
.. [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
1572-1583 DOI: 10.1021/acscentsci.9b00576
"""

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

# add vocab_file dict
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def get_default_tokenizer():
    default_vocab_path = (pkg_resources.resource_filename(
        "deepchem", "feat/tests/vocab.txt"))
    return SmilesTokenizer(default_vocab_path)


class SmilesTokenizer(BertTokenizer):
    """
    Creates the SmilesTokenizer class. The tokenizer heavily inherits from the BertTokenizer
    implementation found in Huggingface's transformers library. It runs a WordPiece tokenization
    algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

    Please see https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp for more details.

    Examples
    --------
    >>> from deepchem.feat.smiles_tokenizer import SmilesTokenizer
    >>> current_dir = os.path.dirname(os.path.realpath(__file__))
    >>> vocab_path = os.path.join(current_dir, 'tests/data', 'vocab.txt')
    >>> tokenizer = SmilesTokenizer(vocab_path)
    >>> print(tokenizer.encode("CC(=O)OC1=CC=CC=C1C(=O)O"))
    [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 16, 22, 16, 16, 22, 16, 20, 16, 17, 22, 19, 18, 19, 13]


    References
    ----------
    .. [1] Schwaller, Philippe; Probst, Daniel; Vaucher, Alain C.; Nair, Vishnu H; Kreutter, David;
        Laino, Teodoro; et al. (2019): Mapping the Space of Chemical Reactions using Attention-Based Neural
        Networks. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.9897365.v3

    Note
    ----
    This class requires huggingface's transformers and tokenizers libraries to be installed.
    """
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file: str = '',
        # unk_token="[UNK]",
        # sep_token="[SEP]",
        # pad_token="[PAD]",
        # cls_token="[CLS]",
        # mask_token="[MASK]",
        **kwargs):
        """Constructs a SmilesTokenizer.

        Parameters
        ----------
        vocab_file: str
            Path to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt
        """

        super().__init__(vocab_file, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocab file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.highest_unused_index = max([
            i for i, v in enumerate(self.vocab.keys())
            if v.startswith("[unused")
        ])
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        self.basic_tokenizer = BasicSmilesTokenizer()

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str, max_seq_length: int = 512, **kwargs):
        """Tokenize a string into a list of tokens.

        Parameters
        ----------
        text: str
            Input string sequence to be tokenized.
        """

        max_len_single_sentence = max_seq_length - 2
        split_tokens = [
            token for token in self.basic_tokenizer.tokenize(text)
            [:max_len_single_sentence]
        ]
        return split_tokens

    def _convert_token_to_id(self, token: str):
        """Converts a token (str/unicode) in an id using the vocab.

        Parameters
        ----------
        token: str
            String token from a larger sequence to be converted to a numerical id.
        """

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (string/unicode) using the vocab.

        Parameters
        ----------
        index: int
            Integer index to be converted back to a string-based token as part of a larger sequence.
        """

        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        """Converts a sequence of tokens (string) in a single string.

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.

        Returns
        -------
        out_string: str
            Single string from combined tokens.
        """

        out_string: str = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self,
                                               token_ids: List[Optional[int]]):
        """Adds special tokens to the a sequence for sequence classification tasks.

        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------
        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        """

        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        """Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.
        """
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(
            self, token_ids_0: List[Optional[int]],
            token_ids_1: List[Optional[int]]) -> List[Optional[int]]:
        """Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

        Parameters
        ----------
        token_ids_0: List[int]
            List of ids for the first string sequence in the sequence pair (A).
        token_ids_1: List[int]
            List of tokens for the second string sequence in the sequence pair (B).
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self,
                           token_ids: List[Optional[int]],
                           length: int,
                           right: bool = True) -> List[Optional[int]]:
        """Adds padding tokens to return a sequence of length max_length.
        By default padding tokens are added to the right of the sequence.

        Parameters
        ----------
        token_ids: list[optional[int]]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        length: int
            TODO
        right: bool, default True
            TODO

        Returns
        -------
        List[int]
            TODO
        """
        padding = [self.pad_token_id] * (length - len(token_ids))

        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ):  # -> Tuple[str]: doctest issue raised with this return type annotation
        """Save the tokenizer vocabulary to a file.

        Parameters
        ----------
        vocab_path: obj: str
            The directory in which to save the SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt

        Returns
        -------
        vocab_file: Tuple
            Paths to the files saved.
            typle with string to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt
        """
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory,
                                      VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(),
                                             key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".
                        format(vocab_file))
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicSmilesTokenizer(object):
    """
    Run basic SMILES tokenization using a regex pattern developed by Schwaller et. al.
    This tokenizer is to be used when a tokenizer that does not require the transformers library by HuggingFace is required.

    Examples
    --------
    >>> from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
    >>> tokenizer = BasicSmilesTokenizer()
    >>> print(tokenizer.tokenize("CC(=O)OC1=CC=CC=C1C(=O)O"))
    ['C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C', '(', '=', 'O', ')', 'O']


    References
    ----------
    .. [1] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
        ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
        1572-1583 DOI: 10.1021/acscentsci.9b00576
    """

    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        """Constructs a BasicSMILESTokenizer.

        Parameters
        ----------
        regex: string
            SMILES token regex
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        """Basic Tokenization of a SMILES.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
