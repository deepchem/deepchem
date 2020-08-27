Tokenizers
===========

A tokenizer is in charge of preparing the inputs for a model. The HuggingFace transformers library (which DeepChem tokenizers are built on top of) comprise tokenizers for all transformer models.

The base classes PreTrainedTokenizer and PreTrainedTokenizerFast implements the common methods for encoding string inputs in model inputs and instantiating/saving python tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library (downloaded from HuggingFace’s AWS S3 repository).

PreTrainedTokenizer and PreTrainedTokenizerFast thus implements the main methods for using all the tokenizers:

- Tokenizing (spliting strings in sub-word token strings), converting tokens strings to ids and back, and encoding/decoding (i.e. tokenizing + convert to integers),

- Adding new tokens to the vocabulary in a way that is independant of the underlying structure (BPE, SentencePiece…),

- Managing special tokens like mask, beginning-of-sentence, etc tokens (adding them, assigning them to attributes in the tokenizer for easy access and making sure they are not split during tokenization)

BatchEncoding holds the output of the tokenizer’s encoding methods (__call__, encode_plus and batch_encode_plus) and is derived from a Python dictionary. When the tokenizer is a pure python tokenizer, this class behave just like a standard python dictionary and hold the various model inputs computed by these methodes (input_ids, attention_mask…).

For more details on the base tokenizers which the DeepChem tokenizers inherit from, please refer to the following: `HuggingFace tokenizers docs <https://huggingface.co/transformers/main_classes/tokenizer.html>`_


SmilesTokenizer
^^^^^^^^^^^^^^^

The :code:`dc.feat.SmilesTokenizer` module inherits from the BertTokenizer class. It runs a WordPiece tokenization algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

References:

-  `RXN Mapper: Unsupervised Attention-Guided Atom-Mapping <https://chemrxiv.org/articles/Unsupervised_Attention-Guided_Atom-Mapping/12298559>`_
-  `Molecular Transformer: Unsupervised Attention-Guided Atom-Mapping <https://pubs.acs.org/doi/10.1021/acscentsci.9b00576>`_

.. autoclass:: deepchem.feat.SmilesTokenizer
  :members: