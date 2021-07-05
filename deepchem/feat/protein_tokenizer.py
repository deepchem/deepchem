from deepchem.feat.base_classes import Featurizer
from typing import Iterable
import numpy as np
import logging
logger = logging.getLogger(__name__)
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

  Usage Example (based on HuggingFace and RostLab ProtBert docs)
  --------------------------------------------------------------
  >>> from deepchem.feat import BertFeaturizer
  >>> featurizer = BertFeaturizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
  >>> inputs = featurizer.featurize('D L I P [MASK] L V T', return_tensors="pt")

  Notes
  -----
  This class is heavily based on and very similar to Ahmad and Chithrananda's
  proposed RobertaFeaturizer wrapper class in DeepChem.

  This class inherits from BertTokenizerFast and Featurizer.
  """

  def __init__(self, input_ids, attention_mask):
    
