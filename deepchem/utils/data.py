"""Utilities for handling datasets."""

import numpy as np
import deepchem as dc


def datasetify(data_like):
  """Convert input into a dataset if possible.

  This utility function attempts to intelligently convert it's
  input into a DeepChem dataset object. Here are the classes of
  common sense transformations it attempts to apply:

  - `dc.data.Dataset`: If the input is already a
  `dc.data.Dataset`, just return unmodified.

  >>> import deepchem as dc
  >>> import numpy as np
  >>> X = np.random.rand(5, 5)
  >>> y = np.random.rand(5,)
  >>> w = np.random.rand(5,)
  >>> ids = np.arange(5)
  >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
  >>> d = datasetify(dataset)

  - List of strings: The strings are assumed to be unique identifiers. They are packaged into `dc.data.NumpyDataset`, as follows.

  >>> l = ["C", "CC"]
  >>> d = datasetify(l)

  - Numpy array: This array is assumed to be the `X` feature array. This is packaged as follows

  >>> d = datasetify(X)

  - Tuple: This is assumed to be a tuple of arrays of form `(X,)` or `(X, y)` or `(X, y, w)` or `(X, y, w, ids)`. This is packaged as follows

  >>> d1 = datasetify((X,))
  >>> d2 = datasetify((X, y))
  >>> d3 = datasetify((X, y, w))
  >>> d4 = datasetify((X, y, w, ids))

  Parameters
  ----------
  data_like: object
    Some object which will attempt to be converted to a
    `dc.data.Dataset` object.

  Returns
  -------
  If successful in conversion, returns `dc.data.NumpyDataset`
  object. Else raises `ValueError`.
  """
  if isinstance(data_like, dc.data.Dataset):
    return data_like
  elif isinstance(data_like, list):
    if len(data_like) > 0 and isinstance(data_like[0], str):
      return dc.data.NumpyDataset(
          X=np.array(data_like), ids=np.array(data_like))
  elif isinstance(data_like, tuple):
    # Assume (X,)
    if len(data_like) == 1:
      return dc.data.NumpyDataset(data_like[0])
    # Assume (X, y)
    elif len(data_like) == 2:
      return dc.data.NumpyDataset(data_like[0], data_like[1])
    # Assume (X, y, z)
    elif len(data_like) == 3:
      return dc.data.NumpyDataset(data_like[0], data_like[1], data_like[2])
    # Assume (X, y, z, ids)
    elif len(data_like) == 4:
      return dc.data.NumpyDataset(data_like[0], data_like[1], data_like[2],
                                  data_like[3])
    else:
      raise ValueError("Cannot convert into Dataset object.")
  elif isinstance(data_like, np.ndarray):
    return dc.data.NumpyDataset(data_like)
  else:
    raise ValueError("Cannot convert into Dataset object.")
