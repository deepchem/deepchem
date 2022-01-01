import deepchem as dc
import numpy as np


def test_setshard_with_X_y():
  """Test setharding on a simple example"""
  X = np.random.rand(10, 3)
  y = np.random.rand(10,)
  dataset = dc.data.DiskDataset.from_numpy(X, y)
  assert dataset.get_shape()[0][0] == 10
  assert dataset.get_shape()[1][0] == 10
  for i, (X, y, w, ids) in enumerate(dataset.itershards()):
    X = X[1:]
    y = y[1:]
    w = w[1:]
    ids = ids[1:]
    dataset.set_shard(i, X, y, w, ids)
  assert dataset.get_shape()[0][0] == 9
  assert dataset.get_shape()[1][0] == 9
