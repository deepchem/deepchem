from os import sep
from deepchem.feat import Featurizer
from typing import List
import numpy as np
try:
    from transformers import RobertaTokenizerFast
except ModuleNotFoundError:
  raise ImportError(
      'Transformers must be installed for RobertaFeaturizer to be used!')
  pass

class RxnFeaturizer(Featurizer):
  def __init__(self, tokenizer: RobertaTokenizerFast, sep_reagent: bool):
    if not isinstance(tokenizer, RobertaTokenizerFast):
      raise TypeError(f"""`tokenizer` must be a constructed `RobertaTokenizerFast`
                       object, not {type(tokenizer)}""")
    else:
      self.tokenizer = tokenizer
    self.sep_reagent = sep_reagent

  def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
    #if dont want to tokenize, return raw reaction SMILES.
    #sep_reagent then tokenize, source and target separately.
    reactant = list(map(lambda x: x.split('>')[0], datapoint))
    reagent = list(map(lambda x: x.split('>')[1], datapoint))
    product = list(map(lambda x: x.split('>')[2], datapoint))

    if self.sep_reagent:
      source = [x + '>' + y for x, y in zip(reactant, reagent)]
    else:
      source = [
          x + '.' + y + '>' if y else x + '>' + y
          for x, y in zip(reactant, reagent)
      ]

    target = product




    return self.tokenizer(source), self.tokenizer(target)