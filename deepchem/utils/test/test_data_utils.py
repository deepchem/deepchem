import unittest
import numpy as np
import deepchem as dc
from deepchem.utils.data import datasetify

class TestDataUtils(unittest.TestCase):

  def test_datasetify(self):
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
