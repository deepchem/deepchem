from deepchem.feat import Featurizer
from typing import List
import numpy as np
from logging import Logger

try:
  from transformers import RobertaTokenizerFast
except ModuleNotFoundError:
  raise ImportError(
      'Transformers must be installed for RxnFeaturizer to be used!')
  pass


class RxnFeaturizer(Featurizer):
  """Reaction Featurizer.

  RxnFeaturizer is a wrapper class for HuggingFace's RobertaTokenizerFast,
  that is intended for featurizing chemical reaction datasets. The featurizer
  computes the source and target required for a seq2seq task and applies the
  RobertaTokenizer on them separately.

  """

  def __init__(self, tokenizer: RobertaTokenizerFast, sep_reagent: bool):
    if not isinstance(tokenizer, RobertaTokenizerFast):
      raise TypeError(
          f"""`tokenizer` must be a constructed `RobertaTokenizerFast`
                       object, not {type(tokenizer)}""")
    else:
      self.tokenizer = tokenizer
    self.sep_reagent = sep_reagent

  def _featurize(self, datapoint: str, **kwargs) -> List[List[List[int]]]:
    # if dont want to tokenize, return raw reaction SMILES.
    # sep_reagent then tokenize, source and target separately.

    datapoint = [datapoint]
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

    source_encoding = list(self.tokenizer(source, padding=True, **kwargs).values())
    target_encoding = list(self.tokenizer(target, padding=True, **kwargs).values())

    return [source_encoding, target_encoding]

def __call__(self, *args, **kwargs) -> np.ndarray:
  return self.featurize(*args, **kwargs)
