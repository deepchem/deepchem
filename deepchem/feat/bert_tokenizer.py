from deepchem.feat.base_classes import Featurizer
from typing import List, Dict
try:
  from transformers import BertTokenizerFast
except:
  raise ImportError("""This class requires the transformers package,
                    which was not found in your environment.""")


class BertFeaturizer(BertTokenizerFast, Featurizer):
  """Bert Featurizer.

  The Bert Featurizer is a wrapper class for HuggingFace's BertTokenizerFast.

  This class intends to allow users to use the BertTokenizer API while
  remaining inside the DeepChem ecosystem.

  Examples
  --------
  >>> from deepchem.feat import BertFeaturizer
  >>> featurizer = BertFeaturizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
  >>> inputs = featurizer.featurize('D L I P [MASK] L V T', return_tensors="pt")

  Notes
  -----
  This class inherits from BertTokenizerFast.

  This class may contain code and/or documentation taken from the
  RobertaFeaturizer pull request (#2581), which have been moved here due to
  code restructuring.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return

  def _featurize(self, sequence: str, **kwargs) -> List[List[int]]:
    """Tokenizes a datapoint with BertTokenizerFast.

    Parameters
    ----------
    sequence: str
        An arbitrary string sequence to be tokenized.

    Returns
    -------
    encoding: list
        list containing two lists: `input_ids` and `attention_mask`
    """
    # print(f"encoding {sequence}")  # Bug: This is getting called 142 times per token
    encoding = list(self(sequence, **kwargs).values())
    return encoding

  def __call__(self, *args, **kwargs) -> Dict[str, List[int]]:
    return super().__call__(*args, **kwargs)
