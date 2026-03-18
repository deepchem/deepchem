from typing import TYPE_CHECKING
from deepchem.feat.base_classes import Featurizer

if TYPE_CHECKING:
    import transformers


class HuggingFaceFeaturizer(Featurizer):
    """Wrapper class that wraps HuggingFace tokenizers as DeepChem featurizers

    The `HuggingFaceFeaturizer` wrapper provides a wrapper
    around Hugging Face tokenizers allowing them to be used as DeepChem
    featurizers. This might be useful in scenarios where user needs to use
    a hugging face tokenizer when loading a dataset.

    Example
    -------
    >>> from deepchem.feat import HuggingFaceFeaturizer
    >>> from transformers import RobertaTokenizerFast
    >>> hf_tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_60k")
    >>> featurizer = HuggingFaceFeaturizer(tokenizer=hf_tokenizer)
    >>> result = featurizer.featurize(['CC(=O)C'])
    """

    def __init__(
        self,
        tokenizer: 'transformers.tokenization_utils_fast.PreTrainedTokenizerFast'
    ):
        """Initializes a tokenizer wrapper

        Parameters
        ----------
        tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast
            The tokenizer to use for featurization
        """
        self.tokenizer = tokenizer

    def _featurize(self, datapoint):
        """Featurizes a single datapoint using the tokenizer"""
        return self.tokenizer(datapoint).data
