"""
Tests for transformer objects.
"""
import os
import unittest
import numpy as np
import pytest
import scipy.ndimage
try:
    import tensorflow as tf
    has_tensorflow = True
except:
    has_tensorflow = False

import deepchem as dc
from deepchem.trans.transformers import DataTransforms


class TestDataTransforms(unittest.TestCase):
    """
    Test DataTransforms for images
    """

    @pytest.mark.tensorflow
    def setUp(self):
        """
        init to load the MNIST data for DataTransforms Tests
        """
        super(TestDataTransforms, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.mnist.load_data()
        train = dc.data.NumpyDataset(x_train, y_train)
        # extract only the images (no need of the labels)
        data = (train.X)[0]
        # reshaping the vector to image
        data = np.reshape(data, (28, 28))
        self.d = data

    @pytest.mark.tensorflow
    def test_blurring(self):
        # Check Blurring
        dt = DataTransforms(self.d)
        blurred = dt.gaussian_blur(sigma=1.5)
        check_blur = scipy.ndimage.gaussian_filter(self.d, 1.5)
        assert np.allclose(check_blur, blurred)

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
    def test_crop(self):
        # Check crop
        dt = DataTransforms(self.d)
        crop = dt.crop(0, 10, 0, 10)
        y = self.d.shape[0]
        x = self.d.shape[1]
        check_crop = self.d[10:y - 10, 0:x - 0]
        assert np.allclose(crop, check_crop)

    @pytest.mark.tensorflow
    def test_convert2gray(self):
        # Check convert2gray
        dt = DataTransforms(self.d)
        gray = dt.convert2gray()
        check_gray = np.dot(self.d[..., :3], [0.2989, 0.5870, 0.1140])
        assert np.allclose(check_gray, gray)

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
    def test_flipping(self):
        # Check flip
        dt = DataTransforms(self.d)
        flip_lr = dt.flip(direction="lr")
        flip_ud = dt.flip(direction="ud")
        check_lr = np.fliplr(self.d)
        check_ud = np.flipud(self.d)
        assert np.allclose(flip_ud, check_ud)
        assert np.allclose(flip_lr, check_lr)

    @pytest.mark.tensorflow
    def test_scaling(self):
        from PIL import Image
        # Check Scales
        dt = DataTransforms(self.d)
        h = 150
        w = 150
        scale = Image.fromarray(self.d).resize((h, w))
        check_scale = dt.scale(h, w)
        np.allclose(scale, check_scale)

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
    def test_median_filter(self):
        # Check median filter
        from PIL import Image, ImageFilter
        dt = DataTransforms(self.d)
        filtered = dt.median_filter(size=3)
        image = Image.fromarray(self.d)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        check_filtered = np.array(image)
        assert np.allclose(check_filtered, filtered)
