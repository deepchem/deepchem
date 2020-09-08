Tokenizers
===========

A tokenizer is in charge of preparing the inputs for a natural language processing model. For many scientific applications, it is possible to treat inputs as "words"/"sentences" and use NLP methods to make meaningful predictions. For example, SMILES strings or DNA sequences have grammatical structure and can be usefully modeled with NLP techniques. DeepChem provides some scientifically relevant tokenizers for use in different applications. These tokenizers are based on those from the Huggingface transformers library (which DeepChem tokenizers inherit from).

The base classes PreTrainedTokenizer and PreTrainedTokenizerFast implements the common methods for encoding string inputs in model inputs and instantiating/saving python tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library (downloaded from HuggingFace’s AWS S3 repository).

PreTrainedTokenizer `(transformers.PreTrainedTokenizer) <https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer>`_ thus implements the main methods for using all the tokenizers:

- Tokenizing (spliting strings in sub-word token strings), converting tokens strings to ids and back, and encoding/decoding (i.e. tokenizing + convert to integers),

- Adding new tokens to the vocabulary in a way that is independant of the underlying structure (BPE, SentencePiece…),

- Managing special tokens like mask, beginning-of-sentence, etc tokens (adding them, assigning them to attributes in the tokenizer for easy access and making sure they are not split during tokenization)

BatchEncoding holds the output of the tokenizer’s encoding methods (__call__, encode_plus and batch_encode_plus) and is derived from a Python dictionary. When the tokenizer is a pure python tokenizer, this class behave just like a standard python dictionary and hold the various model inputs computed by these methodes (input_ids, attention_mask…).

For more details on the base tokenizers which the DeepChem tokenizers inherit from, please refer to the following: `HuggingFace tokenizers docs <https://huggingface.co/transformers/main_classes/tokenizer.html>`_

Tokenization methods on string-based corpuses in the life sciences are becoming increasingly popular for NLP-based applications to chemistry and biology. One such example is ChemBERTa, a transformer for molecular property prediction. DeepChem offers a tutorial for utilizing ChemBERTa using an alternate tokenizer, a Byte-Piece Encoder, which can be found `here. <https://github.com/deepchem/deepchem/blob/master/examples/tutorials/22_Transfer_Learning_With_HuggingFace_tox21.ipynb>`_

SmilesTokenizer
^^^^^^^^^^^^^^^

The :code:`dc.feat.SmilesTokenizer` module inherits from the BertTokenizer class in transformers. It runs a WordPiece tokenization algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

The SmilesTokenizer employs an atom-wise tokenization strategy using the following Regex expression:

>>> SMI_REGEX_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"

To use, please install the transformers package using the following pip command:

>>> pip install transformers

References:

-  `RXN Mapper: Unsupervised Attention-Guided Atom-Mapping <https://chemrxiv.org/articles/Unsupervised_Attention-Guided_Atom-Mapping/12298559>`_
-  `Molecular Transformer: Unsupervised Attention-Guided Atom-Mapping <https://pubs.acs.org/doi/10.1021/acscentsci.9b00576>`_

.. autoclass:: deepchem.feat.SmilesTokenizer
  :members:

BasicSmilesTokenizer
^^^^^^^^^^^^^^^^^^^^

The :code:`dc.feat.BasicSmilesTokenizer` module uses a regex tokenization pattern to tokenise SMILES strings. The regex is developed by Schwaller et. al. The tokenizer is to be used on SMILES in cases where the user wishes to not rely on the transformers API.

References:
-  `Molecular Transformer: Unsupervised Attention-Guided Atom-Mapping <https://pubs.acs.org/doi/10.1021/acscentsci.9b00576>`_

.. autoclass:: deepchem.feat.BasicSmilesTokenizer
  :members:
