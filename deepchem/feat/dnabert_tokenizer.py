from deepchem.feat import Featurizer
from typing import Dict, List
try:
    from transformers import PreTrainedTokenizerFast
except ModuleNotFoundError:
    raise ImportError(
        'Transformers must be installed for DNABertFeaturizer to be used!')
    pass


class DNABertFeaturizer(PreTrainedTokenizerFast, Featurizer):
    """DNABERT-2 Featurizer.

    The DNABertFeaturizer is a wrapper class of the PreTrainedTokenizerFast,
    which is used by Huggingface's transformers library for tokenizing DNA sequences for DNABERT-2 Models.

    Please see https://github.com/huggingface/transformers
    and https://github.com/Zhihan1996/DNABERT_2 for more details.

    Examples
    --------
    DNABertFeaturizer can be used directly as a HuggingFace tokenizer via `__call__`,
    returning a `BatchEncoding` dictionary with `input_ids`, `token_type_ids` and `attention_mask`.

    >>> from deepchem.feat.dnabert_tokenizer import DNABertFeaturizer
    >>> sequences = ["ACGTACGT", "GGGTTTCCC"]
    >>> featurizer = DNABertFeaturizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    >>> out = featurizer(sequences, add_special_tokens=True, truncation=True)
    >>> list(out.keys())
    ['input_ids', 'token_type_ids', 'attention_mask']

    It can also be used as a DeepChem featurizer via the `.featurize()` method,
    which returns a numpy array of shape `(n_sequences, 3, max_length)`.

    >>> from deepchem.feat.dnabert_tokenizer import DNABertFeaturizer
    >>> sequences = ["ACGTACGT", "GGGTTTCCC"]
    >>> featurizer = DNABertFeaturizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    >>> feats = featurizer.featurize(sequences, add_special_tokens=True, truncation=True, padding='max_length', max_length=100)
    >>> feats.shape
    (2, 3, 100)

    Note
    -----
    This class requires transformers to be installed.
    DNABertFeaturizer uses dual inheritance with PreTrainedTokenizerFast in Huggingface for rapid tokenization,
    as well as DeepChem's Featurizer class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
        """Calculate encoding using HuggingFace's PreTrainedTokenizerFast

        Parameters
        ----------
        datapoint: str
            DNA sequence string to be tokenized.

        Returns
        -------
        encoding: List
            List containing three lists; the `input_ids`, `token_type_ids` and the `attention_mask`
        """

        encoding = list(self(datapoint, **kwargs).values())
        return encoding

    def __call__(self, *args, **kwargs) -> Dict[str, List[int]]:
        return super().__call__(*args, **kwargs)
