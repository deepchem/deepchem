"""
Tests for transformer objects.
"""
from deepchem.molnet import load_delaney
from deepchem.trans.transformers import DataTransforms

import os
import unittest
import numpy as np
import pandas as pd
import deepchem as dc
import tensorflow as tf
import scipy.ndimage


def load_solubility_data():
  """Loads solubility dataset"""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  featurizer = dc.feat.CircularFingerprint(size=1024)
  tasks = ["log-solubility"]
  task_type = "regression"
  input_file = os.path.join(current_dir, "../../models/tests/example.csv")
  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)

  return loader.create_dataset(input_file)


class TestTransformers(unittest.TestCase):
  """
  Test top-level API for transformer objects.
  """

  def setUp(self):
    super(TestTransformers, self).setUp()
    self.current_dir = os.path.dirname(os.path.abspath(__file__))
    '''
       init to load the MNIST data for DataTransforms Tests
      '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train = dc.data.NumpyDataset(x_train, y_train)
    # extract only the images (no need of the labels)
    data = (train.X)[0]
    # reshaping the vector to image
    data = np.reshape(data, (28, 28))
    self.d = data

  def test_clipping_X_transformer(self):
    """Test clipping transformer on X of singletask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    ids = np.arange(n_samples)
    X = np.ones((n_samples, n_features))
    target = 5. * X
    X *= 6.
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    transformer = dc.trans.ClippingTransformer(transform_X=True, x_max=5.)
    clipped_dataset = transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (clipped_dataset.X, clipped_dataset.y,
                            clipped_dataset.w, clipped_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check y is unchanged since this is an X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check X is now holding the proper values when sorted.
    np.testing.assert_allclose(X_t, target)

  def test_clipping_y_transformer(self):
    """Test clipping transformer on y of singletask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    ids = np.arange(n_samples)
    X = np.zeros((n_samples, n_features))
    y = np.ones((n_samples, n_tasks))
    target = 5. * y
    y *= 6.
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    transformer = dc.trans.ClippingTransformer(transform_y=True, y_max=5.)
    clipped_dataset = transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (clipped_dataset.X, clipped_dataset.y,
                            clipped_dataset.w, clipped_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
      assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now holding the proper values when sorted.
    np.testing.assert_allclose(y_t, target)

  def test_coulomb_fit_transformer(self):
    """Test coulomb fit transformer on singletask dataset."""
    n_samples = 10
    n_features = 3
    n_tasks = 1
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features, n_features)
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    fit_transformer = dc.trans.CoulombFitTransformer(dataset)
    X_t = fit_transformer.X_transform(dataset.X)
    assert len(X_t.shape) == 2

  def test_IRV_transformer(self):
    n_features = 128
    n_samples = 20
    test_samples = 5
    n_tasks = 2
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.zeros((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids=None)
    X_test = np.random.randint(2, size=(test_samples, n_features))
    y_test = np.zeros((test_samples, n_tasks))
    w_test = np.ones((test_samples, n_tasks))
    test_dataset = dc.data.NumpyDataset(X_test, y_test, w_test, ids=None)
    sims = np.sum(
        X_test[0, :] * X, axis=1, dtype=float) / np.sum(
            np.sign(X_test[0, :] + X), axis=1, dtype=float)
    sims = sorted(sims, reverse=True)
    IRV_transformer = dc.trans.IRVTransformer(10, n_tasks, dataset)
    test_dataset_trans = IRV_transformer.transform(test_dataset)
    dataset_trans = IRV_transformer.transform(dataset)
    assert test_dataset_trans.X.shape == (test_samples, 20 * n_tasks)
    assert np.allclose(test_dataset_trans.X[0, :10], sims[:10])
    assert np.allclose(test_dataset_trans.X[0, 10:20], [0] * 10)
    assert not np.isclose(dataset_trans.X[0, 0], 1.)

  def test_blurring(self):
    # Check Blurring
    dt = DataTransforms(self.d)
    blurred = dt.gaussian_blur(sigma=1.5)
    check_blur = scipy.ndimage.gaussian_filter(self.d, 1.5)
    assert np.allclose(check_blur, blurred)

  def test_center_crop(self):
    # Check center crop
    dt = DataTransforms(self.d)
    x_crop = 50
    y_crop = 50
    crop = dt.center_crop(x_crop, y_crop)
    y = self.d.shape[0]
    x = self.d.shape[1]
    x_start = x // 2 - (x_crop // 2)
    y_start = y // 2 - (y_crop // 2)
    check_crop = self.d[y_start:y_start + y_crop, x_start:x_start + x_crop]
    assert np.allclose(check_crop, crop)

  def test_crop(self):
    #Check crop
    dt = DataTransforms(self.d)
    crop = dt.crop(0, 10, 0, 10)
    y = self.d.shape[0]
    x = self.d.shape[1]
    check_crop = self.d[10:y - 10, 0:x - 0]
    assert np.allclose(crop, check_crop)

  def test_convert2gray(self):
    # Check convert2gray
    dt = DataTransforms(self.d)
    gray = dt.convert2gray()
    check_gray = np.dot(self.d[..., :3], [0.2989, 0.5870, 0.1140])
    assert np.allclose(check_gray, gray)

  def test_rotation(self):
    # Check rotation
    dt = DataTransforms(self.d)
    angles = [0, 5, 10, 90]
    for ang in angles:
      rotate = dt.rotate(ang)
      check_rotate = scipy.ndimage.rotate(self.d, ang)
      assert np.allclose(rotate, check_rotate)

    # Some more test cases for flip
    rotate = dt.rotate(-90)
    check_rotate = scipy.ndimage.rotate(self.d, 270)
    assert np.allclose(rotate, check_rotate)

  def test_flipping(self):
    # Check flip
    dt = DataTransforms(self.d)
    flip_lr = dt.flip(direction="lr")
    flip_ud = dt.flip(direction="ud")
    check_lr = np.fliplr(self.d)
    check_ud = np.flipud(self.d)
    assert np.allclose(flip_ud, check_ud)
    assert np.allclose(flip_lr, check_lr)

  def test_scaling(self):
    from PIL import Image
    # Check Scales
    dt = DataTransforms(self.d)
    h = 150
    w = 150
    scale = Image.fromarray(self.d).resize((h, w))
    check_scale = dt.scale(h, w)
    np.allclose(scale, check_scale)

  def test_shift(self):
    # Check shift
    dt = DataTransforms(self.d)
    height = 5
    width = 5
    if len(self.d.shape) == 2:
      shift = scipy.ndimage.shift(self.d, [height, width])
    if len(self.d.shape) == 3:
      shift = scipy.ndimage.shift(self.d, [height, width, 0])
    check_shift = dt.shift(width, height)
    assert np.allclose(shift, check_shift)

  def test_gaussian_noise(self):
    # check gaussian noise
    dt = DataTransforms(self.d)
    np.random.seed(0)
    random_noise = self.d
    random_noise = random_noise + np.random.normal(
        loc=0, scale=25.5, size=self.d.shape)
    np.random.seed(0)
    check_random_noise = dt.gaussian_noise(mean=0, std=25.5)
    assert np.allclose(random_noise, check_random_noise)

  def test_salt_pepper_noise(self):
    # check salt and pepper noise
    dt = DataTransforms(self.d)
    np.random.seed(0)
    prob = 0.05
    random_noise = self.d
    noise = np.random.random(size=self.d.shape)
    random_noise[noise < (prob / 2)] = 0
    random_noise[noise > (1 - prob / 2)] = 255
    np.random.seed(0)
    check_random_noise = dt.salt_pepper_noise(prob, salt=255, pepper=0)
    assert np.allclose(random_noise, check_random_noise)

  def test_median_filter(self):
    #Check median filter
    from PIL import Image, ImageFilter
    dt = DataTransforms(self.d)
    filtered = dt.median_filter(size=3)
    image = Image.fromarray(self.d)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    check_filtered = np.array(image)
    assert np.allclose(check_filtered, filtered)
