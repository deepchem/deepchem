import numpy as np
import deepchem as dc
from deepchem.utils.data import datasetify


def test_datasetify():
  l = ["C", "CC"]
  d = datasetify(l)
  assert isinstance(d, dc.data.Dataset)
  assert (d.ids == np.array(l)).all()

  X = np.random.rand(5, 5)
  d = datasetify(X)
  assert (d.X == X).all()

  dataset = dc.data.NumpyDataset(np.random.rand(5, 5))
  d = datasetify(dataset)
  assert (d.X == dataset.X).all()


def test_from_numpy_tuples():
  X = np.random.rand(5, 5)
  y = np.random.rand(5,)
  w = np.random.rand(5,)
  ids = np.arange(5)

  d = datasetify((X,))
  assert (d.X == X).all()

  d = datasetify((X, y))
  assert (d.X == X).all()
  assert (d.y == y).all()

  d = datasetify((X, y, w))
  assert (d.X == X).all()
  assert (d.y == y).all()
  assert (d.w == w).all()

  d = datasetify((X, y, w, ids))
  assert (d.X == X).all()
  assert (d.y == y).all()
  assert (d.w == w).all()
  assert (d.ids == ids).all()
