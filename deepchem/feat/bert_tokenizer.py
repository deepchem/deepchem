from deepchem.feat import Featurizer
from typing import Optional, List, Union
try:
  import transformers
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

  def __init__(self,
               tokenizer: BertTokenizerFast,
               return_tensors: Optional[Union[
                   str, transformers.file_utils.TensorType]] = None):
    """
    Initialize a BertFeaturizer object

    Parameters
    ----------
    tokenizer: BertTokenizerFast
      Tokenizer to be used for featurization.
    return_tensors: Optional[Union[str, transformers.file_utils.TensorType]]
      HuggingFace argument to "return tensors instead of python integers."
      src: https://huggingface.co/transformers/internal/tokenization_utils.html
    """

    if not isinstance(tokenizer, BertTokenizerFast):
      raise TypeError(f"""`tokenizer` must be a constructed `BertTokenizerFast`
                       object, not {type(tokenizer)}""")
    else:
      self.tokenizer = tokenizer
    self.return_tensors = return_tensors

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
    encoding = self.tokenizer(
        datapoint, return_tensors=self.return_tensors, **kwargs)
    encoding = list(encoding.values())
    return encoding
