from deepchem.feat.base_classes import Featurizer
from deepchem.feat.molecule_featurizers import RobertaFeaturizer
from typing import Iterable
import numpy as np
import logging, warnings
logger = logging.getLogger(__name__)
try:
  from transformers import BertTokenizerFast, RobertaTokenizerFast
except:
  raise ImportError("""This class requires the transformers package,
                    which was not found in your environment.""")
# TODO discuss RDKitMol support (perhaps RobertaFeaturizer should not be a moleule featurizer?
# Current approach to handling both molecules and strings is a bit messy.
try:
  from deepchem.utils.typing import RDKitMol
except:
  warnings.warn("""RDKitMol was not found in your environment, so this class
                will only featurize strings and will not featurize RDKitMol
                objects.""")
                

class BertFeaturizer(RobertaFeaturizer, BertTokenizerFast):
  """Bert Featurizer.

  The Bert Featurizer is a wrapper class for HuggingFace's BertTokenizerFast.

  This class intends to allow users to use the BertTokenizer API while
  remaining inside the DeepChem ecosystem.

  Usage Example (based on HuggingFace and RostLab ProtBert docs)
  --------------------------------------------------------------
  >>> from deepchem.feat import BertFeaturizer
  >>> featurizer = BertFeaturizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
  >>> inputs = featurizer.featurize('D L I P [MASK] L V T', return_tensors="pt")

  Notes
  -----
  This class inherits first from RobertaFeaturizer and then from
  BertTokenizerFast. Its logic relies on super() calls in RobertaFeaturizer
  being delegated to BertTokenizerFast in accordance with Python's multiple
  inheritance rules.

  Parts of this class, including its documentation and naming scheme, is based
  on and/or bear similarities to RobertaFeaturizer for the sake of consistency.
  """

  def _featurize(self, sequence: Union[str, RDKitMol]) -> List[List[int]]:
    """Tokenizes a datapoint with BertTokenizerFast.

      Parameters
      ----------
      sequence: Union[str, RDKitMol]
        An arbitrary string or an RDKitMol object. The latter is only supported
        if RDKitMol is imported.

      Returns
      -------
      encoding: list 
        list containing two lists: `input_ids` and `attention_mask`
      """
      if not isinstance(sequence, str):
        super()
      obj = self(sequence, self.input_ids, self.attention_mask)
      encoding = list(obj.values())  # TODO check if list() is extrenuous
      return encoding
