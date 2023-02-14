from deepchem.feat import Featurizer
from typing import Dict, List
try:
    from transformers import RobertaTokenizerFast
except ModuleNotFoundError:
    raise ImportError(
        'Transformers must be installed for RobertaFeaturizer to be used!')
    pass


class RobertaFeaturizer(RobertaTokenizerFast, Featurizer):
    """Roberta Featurizer.

    The Roberta Featurizer is a wrapper class of the Roberta Tokenizer,
    which is used by Huggingface's transformers library for tokenizing large corpuses for Roberta Models.
    Please confirm the details in [1]_.

    Please see https://github.com/huggingface/transformers
    and https://github.com/seyonechithrananda/bert-loves-chemistry for more details.

    Examples
    --------
    >>> from deepchem.feat import RobertaFeaturizer
    >>> smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    >>> featurizer = RobertaFeaturizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
    >>> out = featurizer.featurize(smiles, add_special_tokens=True, truncation=True)

    References
    ----------
    .. [1] Chithrananda, Seyone, Grand, Gabriel, and Ramsundar, Bharath (2020): "Chemberta: Large-scale self-supervised
        pretraining for molecular property prediction." arXiv. preprint. arXiv:2010.09885.


    Note
    -----
    This class requires transformers to be installed.
    RobertaFeaturizer uses dual inheritance with RobertaTokenizerFast in Huggingface for rapid tokenization,
    as well as DeepChem's MolecularFeaturizer class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
        """Calculate encoding using HuggingFace's RobertaTokenizerFast

        Parameters
        ----------
        sequence: str
            Arbitrary string sequence to be tokenized.

        Returns
        -------
        datapoint: List
            List containing two lists; the `input_ids` and the `attention_mask`
        """

        # the encoding is natively a dictionary with keys 'input_ids' and 'attention_mask'
        encoding = list(self(datapoint, **kwargs).values())
        return encoding

    def __call__(self, *args, **kwargs) -> Dict[str, List[int]]:
        return super().__call__(*args, **kwargs)
