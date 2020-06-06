from unittest import TestCase

import deepchem
import numpy as np
from sklearn.model_selection import train_test_split
from deepchem.splits import SpecifiedIndexSplitter


class TestSpecifiedIndexSplitter(TestCase):

  def create_dataset(self):
    n_samples, n_features = 20, 10
    X = np.random.random(size=(n_samples, n_features))
    y = np.random.random(size=(n_samples, 1))
    return deepchem.data.NumpyDataset(X, y)

  def test_split(self):
    ds = self.create_dataset()
    indexes = list(range(len(ds)))
    train, test = train_test_split(indexes)
    train, valid = train_test_split(train)

    splitter = SpecifiedIndexSplitter(train, valid, test)
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(ds)

    self.assertTrue(np.all(train_ds.X == ds.X[train]))
    self.assertTrue(np.all(valid_ds.X == ds.X[valid]))
    self.assertTrue(np.all(test_ds.X == ds.X[test]))
