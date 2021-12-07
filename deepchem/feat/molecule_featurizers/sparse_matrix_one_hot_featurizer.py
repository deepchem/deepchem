import logging
from typing import List

import numpy as np
import scipy 
from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import Featurizer
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

ZINC_CHARSET = [
    '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=',
    '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o',
    'n', 'p', 's', 'r'
]
codes = [
      'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
      'S', 'T', 'V', 'W', 'Y','X','Z','B','U','O'
]

class SparseMatrixOneHotFeaturizer(Featurizer):
  """Encodes any arbitrary string as a one-hot array .

  Standalone Usage:

  >>> import deepchem as dc
  >>> codes = [
      'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
      'S', 'T', 'V', 'W', 'Y','X','Z','B','U','O'
      ]
  >>> featurizer = dc.feat.SparseMatrixOneHotFeaturizer(codes)
  >>> sequence = "MMMQLA"
  >>> encodings = featurizer.featurize(sequence)
  >>> print(encodings)
  (0, 10)	1.0
  (1, 10)	1.0
  (2, 10)	1.0
  (3, 13)	1.0
  (4, 9)	1.0
  (5, 0)	1.0
  >>> encodings[0].shape
  (6, 25)
  >>> featurizer.untransform(encodings[0])
  'MMMQLA'

  """

  def __init__(self,
               charset: List[str] = codes,
               ):
    """Initialize featurizer.

    Parameters
    ----------
    charset: List[str] (default ZINC_CHARSET)
      A list of strings, where each string is length 1 and unique.
    max_length: Optional[int], optional (default 100)
      The max length for string. If the length of string is shorter than
      max_length, the string is padded using space.

      If max_length is None, no padding is performed and arbitrary length
      strings are allowed.
    """
    if len(charset) != len(set(charset)):
      raise ValueError("All values in charset must be unique.")
    self.charset = charset
    from sklearn.preprocessing import OneHotEncoder 
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(self.charset)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    cat = np.array(self.charset).reshape(1,len(self.charset))
    self.ohe = OneHotEncoder(categories = list(cat))
    

  def featurize(self,
                datapoints: Iterable[Any],
                log_every_n: int = 1000,
                **kwargs) -> np.ndarray:
    """Featurize strings.

    Parameters
    ----------
    datapoints: list
      A list of either strings (str or numpy.str_) 
    log_every_n: int, optional (default 1000)
      How many elements are featurized every time a featurization is logged.
    """
    datapoints = list(datapoints)
    if (len(datapoints) < 1):
      return np.array([])
    # Featurize data using featurize() in parent class
    return Featurizer.featurize(self, datapoints, log_every_n)

  def _featurize(self, datapoint: Any, **kwargs):
    # Featurize str data
    if isinstance(datapoint, (str, np.str_)):
      return self._featurize_string(datapoint)


  def _featurize_string(self, string: str) -> scipy.sparse:
    """Compute one-hot featurization of string.

    Parameters
    ----------
    string: str
      An arbitrary string to be featurized.

    Returns
    -------
    scipy.sparse using a OneHotEncoder of Sklearn

    """

    sparse_mat = self.ohe.fit_transform(np.array(list(string)).reshape(-1,1))
    return sparse_mat


  def untransform(self, one_hot_vectors: scipy.sparse) -> str:
    """Convert from one hot representation back to original string

    Parameters
    ----------
    one_hot_vectors: np.ndarray
      An array of one hot encoded features.

    Returns
    -------
    str
      Original string for an one hot encoded array.
    """
    string = ""
    invers_trans = self.ohe.inverse_transform(one_hot_vectors)
    for one_hot in invers_trans:
      string += one_hot  
    return string
