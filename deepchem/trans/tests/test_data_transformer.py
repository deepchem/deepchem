"""
Tests for DataTransformer object.
"""
import os
import unittest
import numpy as np
import pytest
try:
    import tensorflow as tf
    has_tensorflow = True
except:
    has_tensorflow = False

import deepchem as dc
from deepchem.trans.transformers import DataTransformer


class TestDataTransformer(unittest.TestCase):
    """
    Test DataTransformer for image augmentations
    """

    @pytest.mark.tensorflow
    def setUp(self):
        """
        init to load the MNIST data for DataTransformer Tests
        """
        super(TestDataTransformer, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.mnist.load_data()
        train = dc.data.NumpyDataset(x_train, y_train)
        # use a small batch of 5 images
        self.X = (train.X)[:5].astype(np.float64)
        self.y = y_train[:5]
        self.w = np.ones(5)
        self.ids = np.arange(5)

    @pytest.mark.tensorflow
    def test_probability(self):
        # Check that probability=0.0 leaves all images unchanged
        args = {
            "transform_type": "flip",
            "transform_params": {
                "direction": "lr"
            },
            "probability": 0.0,
        }
        dt = DataTransformer(arguments=args)
        X_out, _, _, _ = dt.transform_array(self.X.copy(), self.y, self.w,
                                            self.ids)
        assert np.allclose(X_out, self.X)

    @pytest.mark.tensorflow
    def test_rotation(self):
        # Check that rotation preserves original image shape
        args = {
            "transform_type": "rotate",
            "transform_params": {
                "angle": 45
            },
            "probability": 1.0,
        }
        dt = DataTransformer(arguments=args)
        X_out, _, _, _ = dt.transform_array(self.X.copy(), self.y, self.w,
                                            self.ids)
        assert X_out.shape == self.X.shape

    @pytest.mark.tensorflow
    def test_salt_pepper_noise(self):
        # Check that salt_pepper_noise does not mutate the original array
        args = {
            "transform_type": "salt_pepper_noise",
            "transform_params": {
                "prob": 0.05,
                "salt": 255,
                "pepper": 0
            },
            "probability": 1.0,
        }
        X_original = self.X.copy()
        dt = DataTransformer(arguments=args)
        dt.transform_array(self.X, self.y, self.w, self.ids)
        assert np.allclose(self.X, X_original)

    @pytest.mark.tensorflow
    def test_invalid_transform(self):
        # Check that unsupported transform types raise ValueError
        invalid_types = ["scale", "crop", "center_crop", "convert2gray"]
        for t_type in invalid_types:
            dt = DataTransformer(arguments={"transform_type": t_type})
            with self.assertRaises(ValueError):
                dt.transform_array(self.X.copy(), self.y, self.w, self.ids)
