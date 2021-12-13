from deepchem.feat import Featurizer
from typing import List
try:
  from transformers import BertTokenizerFast
except ModuleNotFoundError:
  raise ImportError(
      'Transformers must be installed for BertFeaturizer to be used!')
  pass


class BertFeaturizer(Featurizer):
  """Bert Featurizer.

  Bert Featurizer.
  The Bert Featurizer is a wrapper class for HuggingFace's BertTokenizerFast.
  This class intends to allow users to use the BertTokenizer API while
  remaining inside the DeepChem ecosystem.

  Examples
  --------
  >>> from deepchem.feat import BertFeaturizer
  >>> from transformers import BertTokenizerFast
  >>> tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
  >>> featurizer = BertFeaturizer(tokenizer)
  >>> feats = featurizer.featurize('D L I P [MASK] L V T')

  Notes
  -----
  Examples are based on RostLab's ProtBert documentation.
  """

  def __init__(self, tokenizer: BertTokenizerFast):
    if not isinstance(tokenizer, BertTokenizerFast):
      raise TypeError(f"""`tokenizer` must be a constructed `BertTokenizerFast`
                       object, not {type(tokenizer)}""")
    else:
      self.tokenizer = tokenizer

  def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
    """
    Calculate encoding using HuggingFace's RobertaTokenizerFast

    Parameters
    ----------
    datapoint: str
      Arbitrary string sequence to be tokenized.

    Returns
    -------
    encoding: List
      List containing three lists: the `input_ids`, 'token_type_ids', and `attention_mask`.
    """

    # the encoding is natively a dictionary with keys 'input_ids', 'token_type_ids', and 'attention_mask'
    encoding = list(self.tokenizer(datapoint, **kwargs).values())
    return encoding
