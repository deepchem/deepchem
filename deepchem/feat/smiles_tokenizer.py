# Requriments - transformers, tokenizers
# Right now, the Smiles Tokenizer uses an exiesting vocab file from rxnfp that is fairly comprehensive and from the USPTO dataset.
# The vocab may be expanded in the near future

import collections
import logging
import os
import re
import numpy as np
import pkg_resources
from typing import List
from transformers import BertTokenizer

# export
SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

def get_default_tokenizer():
    default_vocab_path = (
        pkg_resources.resource_filename(
            "chemberta",
            "tokenizers/vocab.txt"
        )
    )
    return SmilesTokenizer(default_vocab_path)

class SmilesTokenizer(BertTokenizer):
    r"""
    Constructs a SmilesTokenizer.
    Bulk of code is from https://github.com/huggingface/transformers and https://github.com/rxn4chemistry/rxnfp
    Args:
        vocab_file: Path to a SMILES character per line vocabulary file
    """

    def __init__(
            self,
            vocab_file='',
            # unk_token="[UNK]",
            # sep_token="[SEP]",
            # pad_token="[PAD]",
            # cls_token="[CLS]",
            # mask_token="[MASK]",
            **kwargs
    ):
        """Constructs a BertTokenizer.
        Args:
            **vocab_file**: Path to a SMILES character per line vocabulary file
        """
        super().__init__(vocab_file, **kwargs)
        # take into account special tokens in max length
        self.max_len_single_sentence = self.max_len - 2
        self.max_len_sentences_pair = self.max_len - 3

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocab file at path '{}'.".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.highest_unused_index = max(
            [
                i for i, v in enumerate(self.vocab.keys())
                if v.startswith("[unused")
            ]
        )
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.basic_tokenizer = BasicSmilesTokenizer()
        self.init_kwargs["max_len"] = self.max_len

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text):
        split_tokens = [token for token in self.basic_tokenizer.tokenize(text)]
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self, token_ids):
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens):
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_sequence_pair(self, token_0, token_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        sep = [self.sep_token]
        cls = [self.cls_token]
        return cls + token_0 + sep + token_1 + sep

    def add_special_tokens_ids_sequence_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self, token_ids, length, right=True):
        """
        Adds padding tokens to return a sequence of length max_length.
        By  default padding tokens are added to the right of the sequence.
        """
        padding = [self.pad_token_id] * (length - len(token_ids))
        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a file."""
        index = 0
        vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(
                    self.vocab.items(), key=lambda kv: kv[1]
            ):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".
                            format(vocab_file)
                    )
                    index = token_index
                writer.write(token + u"\n")
                index += 1
        return (vocab_file,)

class BasicSmilesTokenizer(object):
    """Run basic SMILES tokenization"""

    def __init__(self, regex_pattern=SMI_REGEX_PATTERN):
        """ Constructs a BasicSMILESTokenizer.
        Args:
            **regex**: SMILES token regex
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        """ Basic Tokenization of a SMILES.
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