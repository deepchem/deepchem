from deepchem.feat import Featurizer
from typing import List
try:
  from transformers import RobertaTokenizerFast
except ModuleNotFoundError:
  raise ImportError(
      'Transformers must be installed for RobertaFeaturizer to be used!')
  pass


class RobertaFeaturizer(Featurizer):
  """Roberta Featurizer.

  The Roberta Featurizer is a wrapper class of the Roberta Tokenizer,
  which is used by Huggingface's transformers library for tokenizing large corpuses for Roberta Models.
  This class intends to allow users to use the RobertaTokenizer API while
  remaining inside the DeepChem ecosystem.

  Please see https://github.com/huggingface/transformers
  and https://github.com/seyonechithrananda/bert-loves-chemistry for more details.

  Examples
  --------
  >>> from transformers import RobertaTokenizerFast
  >>> from deepchem.feat import RobertaFeaturizer
  >>> tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k", do_lower_case=False)
  >>> featurizer = RobertaFeaturizer(tokenizer)
  >>> smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
  >>> out = featurizer.featurize(smiles, add_special_tokens=True, truncation=True)

  References
  ----------
  .. [1] Chithrananda, Seyone, Grand, Gabriel, and Ramsundar, Bharath (2020): "Chemberta: Large-scale self-supervised
    pretraining for molecular property prediction." arXiv. preprint. arXiv:2010.09885.


  Note
  -----
  This class requires transformers to be installed.
  RobertaFeaturizer uses inheritance with DeepChem's 
  Featurizer class while instantiating RobertaTokenizerFast 
  in Huggingface as an attribute for rapid tokenization.
  """

  def __init__(self, tokenizer: RobertaTokenizerFast):
    if not isinstance(tokenizer, RobertaTokenizerFast):
      raise TypeError(f"""`tokenizer` must be a constructed `RobertaTokenizerFast`
                       object, not {type(tokenizer)}""")
    else:
      self.tokenizer = tokenizer

  def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
    """Calculate encoding using HuggingFace's RobertaTokenizerFast

    Parameters
    ----------
    datapoint: str
      Arbitrary string sequence to be tokenized.

    Returns
    -------
    encoding: List
      List containing two lists; the `input_ids` and the `attention_mask`
    """

    # the encoding is natively a dictionary with keys 'input_ids' and 'attention_mask'
    encoding = list(self(datapoint, **kwargs).values())
    return encoding
