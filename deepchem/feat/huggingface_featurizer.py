from deepchem.feat.base_classes import Featurizer


class HuggingFaceFeaturizer(Featurizer):
    """Wrapper class that wraps HuggingFace tokenizers as DeepChem featurizers

    The `HuggingFaceFeaturizer` wrapper model provides a wrapper
    around Hugging Face tokenizers allowing them to be used as DeepChem
    featurizers. This might be useful in scenarios where user needs to use
    a hugging face tokenizer when loading a dataset.

    Example
    -------
    >>> from deepchem.feat import HuggingFaceFeaturizer
    >>> from transformers import RobertaTokenizerFast
    >>> hf_tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_160kk")
    >>> featurizer = HuggingFaceFeaturizer(tokenizer=hf_tokenizer)
    >>> result = featurizer.featurize(['CC(=O)C'])
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _featurize(self, datapoint):
        return self.tokenizer(datapoint).data
