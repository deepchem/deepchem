"""Utilities for handling datasets."""
import numpy as np
import deepchem as dc

def datasetify(data_like):
  """

  This utility function attempts to intelligently convert it's
  input into a DeepChem dataset object. Here are the classes of
  common sense transformations it attempts to apply:

  - `dc.data.Dataset`: If the input is already a
  `dc.data.Dataset`, just return unmodified.
  - List of strings: The strings are assumed to be unique identifiers. They are packaged into `dc.data.NumpyDataset`, as follows.

  >>> import deepchem as dc
  >>> import numpy as np
  >>> l = ["C", "CC"]
  >>> dc.data.NumpyDataset(X=np.array(l), ids=np.array(l))

  The double packaging as `X` and `ids` is awkward, but it's
  currently not feasible to create a `dc.data.NumpyDataset`
  without `X` specified.

  - Numpy array: This array is assumed to be the `X` feature array. This is packaged as follows

  >>> import deepchem as dc
  >>> import numpy as np
  >>> X = np.random.rand(5, 5)
  >>> dc.data.NumpyDataset(X)

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
      return dc.data.NumpyDataset(X=np.array(data_like), ids=np.array(data_like))
  elif isinstance(data_like, np.ndarray):
    return dc.data.NumpyDataset(data_like)
  else:
    raise ValueError("Cannot convert into Dataset object.")

