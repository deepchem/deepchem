from deepchem.feat import Featurizer
from typing import List
import numpy as np

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
  RobertaTokenizer on them separately. Additionally, it can also separate or
  mix the reactants and reagents before tokenizing.

  """

  def __init__(self, tokenizer: RobertaTokenizerFast, sep_reagent: bool):
    """Initialises the featurizer with an appropriate HuggingFace Tokenizer."""
    if not isinstance(tokenizer, RobertaTokenizerFast):
      raise TypeError(
          f"""`tokenizer` must be a constructed `RobertaTokenizerFast`
                       object, not {type(tokenizer)}""")
    else:
      self.tokenizer = tokenizer
    self.sep_reagent = sep_reagent

  def _featurize(self, datapoint: str, **kwargs) -> List[List[List[int]]]:
    """Featurizes a datapoint.

    Processes each entry in the dataset by first applying the reactant-reagent
    mixing, the source/target separation and then the pretrained tokenizer on the
    separated strings.

    """

    datapoint_list = [datapoint]
    reactant = list(map(lambda x: x.split('>')[0], datapoint_list))
    reagent = list(map(lambda x: x.split('>')[1], datapoint_list))
    product = list(map(lambda x: x.split('>')[2], datapoint_list))

    if self.sep_reagent:
      source = [x + '>' + y for x, y in zip(reactant, reagent)]
    else:
      source = [
          x + '.' + y + '>' if y else x + '>' + y
          for x, y in zip(reactant, reagent)
      ]
    target = product

    source_encoding = list(
        self.tokenizer(source, padding=True, **kwargs).values())
    target_encoding = list(
        self.tokenizer(target, padding=True, **kwargs).values())

    return [source_encoding, target_encoding]

  def __call__(self, *args, **kwargs) -> np.ndarray:
    return self.featurize(*args, **kwargs)

  def __str__(self) -> str:
    """Handles file name error.

    Overrides the __str__ method of the Featurizer base class to avoid errors
    while saving the dataset, due to the large default name of the HuggingFace
    tokenizer.
    """
    return 'RxnFeaturizer_' + str(self.sep_reagent)
