from deepchem.feat import Featurizer
from typing import List
try:
    from transformers import RobertaTokenizerFast
except ModuleNotFoundError:
  raise ImportError(
      'Transformers must be installed for RobertaFeaturizer to be used!')
  pass

class RxnFeaturizer(Featurizer):
  def __init__(self, tokenizer: RobertaTokenizerFast):
    if not isinstance(tokenizer, RobertaTokenizerFast):
      raise TypeError(f"""`tokenizer` must be a constructed `RobertaTokenizerFast`
                       object, not {type(tokenizer)}""")
    else:
      self.tokenizer = tokenizer

  def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
    #if dont want to tokenize, return raw reaction SMILES.
    #sep_reagent then tokenize, source and target separately.
    source, target = datapoint.split('>')

    return self.tokenizer(source), self.tokenizer(target)
