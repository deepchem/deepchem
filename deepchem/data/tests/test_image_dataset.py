"""
Tests for ImageDataset class
"""
import unittest
import numpy as np
import deepchem as dc
import os
from tensorflow.python.framework import test_util


class TestImageDataset(test_util.TensorFlowTestCase):
  """
  Test ImageDataset class.
  """

  def test_load_images(self):
    """Test that ImageDataset loads images."""

    path = os.path.join(os.path.dirname(__file__), 'images')
    files = [os.path.join(path, f) for f in os.listdir(path)]

    # First try using images for X.

    ds1 = dc.data.ImageDataset(files, np.random.random(10))
    x_shape, y_shape, w_shape, ids_shape = ds1.get_shape()
    np.testing.assert_array_equal([10, 28, 28], x_shape)
    np.testing.assert_array_equal([10], y_shape)
    np.testing.assert_array_equal([10], w_shape)
    np.testing.assert_array_equal([10], ids_shape)
    np.testing.assert_array_equal(ds1.X.shape, x_shape)
    np.testing.assert_array_equal(ds1.y.shape, y_shape)
    np.testing.assert_array_equal(ds1.w.shape, w_shape)
    np.testing.assert_array_equal(ds1.ids.shape, ids_shape)

    # Now try using images for y.

    ds2 = dc.data.ImageDataset(np.random.random(10), files)
    x_shape, y_shape, w_shape, ids_shape = ds2.get_shape()
    np.testing.assert_array_equal([10], x_shape)
    np.testing.assert_array_equal([10, 28, 28], y_shape)
    np.testing.assert_array_equal([10, 1], w_shape)
    np.testing.assert_array_equal([10], ids_shape)
    np.testing.assert_array_equal(ds2.X.shape, x_shape)
    np.testing.assert_array_equal(ds2.y.shape, y_shape)
    np.testing.assert_array_equal(ds2.w.shape, w_shape)
    np.testing.assert_array_equal(ds2.ids.shape, ids_shape)
    np.testing.assert_array_equal(ds1.X, ds2.y)

  def test_itersamples(self):
    """Test iterating samples of an ImageDataset."""

    path = os.path.join(os.path.dirname(__file__), 'images')
    files = [os.path.join(path, f) for f in os.listdir(path)]
    ds = dc.data.ImageDataset(files, np.random.random(10))
    X = ds.X
    i = 0
    for x, y, w, id in ds.itersamples():
      np.testing.assert_array_equal(x, X[i])
      assert y == ds.y[i]
      assert w == ds.w[i]
      assert id == ds.ids[i]
      i += 1
    assert i == 10

  def test_iterbatches(self):
    """Test iterating batches of an ImageDataset."""

    path = os.path.join(os.path.dirname(__file__), 'images')
    files = [os.path.join(path, f) for f in os.listdir(path)]
    ds = dc.data.ImageDataset(files, np.random.random(10))
    X = ds.X
    iterated_ids = set()
    for x, y, w, ids in ds.iterbatches(2, epochs=2):
      np.testing.assert_array_equal([2, 28, 28], x.shape)
      np.testing.assert_array_equal([2], y.shape)
      np.testing.assert_array_equal([2], w.shape)
      np.testing.assert_array_equal([2], ids.shape)
      for i in (0, 1):
        assert ids[i] in files
        if len(iterated_ids) < 10:
          assert ids[i] not in iterated_ids
          iterated_ids.add(ids[i])
        else:
          assert ids[i] in iterated_ids
        index = files.index(ids[i])
        np.testing.assert_array_equal(x[i], X[index])
    assert len(iterated_ids) == 10


if __name__ == "__main__":
  unittest.main()
