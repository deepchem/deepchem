import logging
from typing import List
import numpy as np
import scipy
from deepchem.feat.base_classes import Featurizer
from typing import Any, Iterable

logger = logging.getLogger(__name__)
CHARSET = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', 'X', 'Z', 'B', 'U', 'O'
]


class SparseMatrixOneHotFeaturizer(Featurizer):
  """Encodes any arbitrary string as a one-hot array.

  This featurizer uses the sklearn OneHotEncoder to create
  sparse matrix representation of a one-hot array of any string.
  It is expected to be used in large datasets that produces memory overload
  using standard featurizer such as OneHotFeaturizer. For example: SwissprotDataset


  Standalone Usage:

  >>> import deepchem as dc
  >>> featurizer = dc.feat.SparseMatrixOneHotFeaturizer()
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
               charset: List[str] = CHARSET):
    """Initialize featurizer.

    Parameters
    ----------
    charset: List[str] (default code)
      A list of strings, where each string is length 1 and unique.
    """
    if len(charset) != len(set(charset)):
      raise ValueError("All values in charset must be unique.")
    self.charset = charset
    from sklearn.preprocessing import OneHotEncoder
    cat = np.array(self.charset).reshape(1, len(self.charset))
    self.ohe = OneHotEncoder(categories=list(cat), handle_unknown='ignore')


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
    #datapoints = datapoints)
    if (len(datapoints) < 1):
      return np.array([])
    # Featurize data using featurize() in parent class
    return Featurizer.featurize(self, datapoints, log_every_n)

  def _featurize(self, datapoint: Any, **kwargs):
      """ Use parent method of base clase Featurizer.
      
      Parameters
      ----------
      datapoint : Any
          DESCRIPTION.
      **kwargs : TYPE
          DESCRIPTION.

      Returns
      -------
      TYPE
          DESCRIPTION.

      """
    # Featurize str data
      if isinstance(datapoint, (str, np.str_)):
        sequence = np.array(list(datapoint)).reshape(-1,1)
        sparse_mat = self.ohe.fit_transform(sequence)
        return  sparse_mat    #self._featurize_string(datapoint)
      else:
        raise ValueError("Datapoint is not a string")

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
