import deepchem as dc
import numpy as np
import pytest
import unittest
from flaky import flaky
try:
    import torch  # noqa: F401
    has_torch = True
except:
    has_torch = False


class TestCNN(unittest.TestCase):

    @pytest.mark.torch
    def test_1d_cnn_regression(self):
        """Test that a 1D CNN can overfit simple regression datasets."""
        n_samples = 10
        n_features = 3
        n_tasks = 1

        np.random.seed(123)
        X = np.random.rand(n_samples, 10, n_features)
        y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
        dataset = dc.data.NumpyDataset(X, y)

        regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
        model = dc.models.CNN(n_tasks,
                              n_features,
                              dims=1,
                              dropouts=0,
                              kernel_size=3,
                              mode='regression',
                              learning_rate=0.003)

        # Fit trained model
        model.fit(dataset, nb_epoch=200)

        # Eval model on train
        scores = model.evaluate(dataset, [regression_metric])
        assert scores[regression_metric.name] < 0.1

    @pytest.mark.torch
    def test_2d_cnn_classification(self):
        """Test that a 2D CNN can overfit simple classification datasets."""
        n_samples = 10
        n_features = 3
        n_tasks = 1

        np.random.seed(123)
        X = np.random.rand(n_samples, 10, 10, n_features)
        y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
        dataset = dc.data.NumpyDataset(X, y)

        classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        model = dc.models.CNN(n_tasks,
                              n_features,
                              dims=2,
                              dropouts=0,
                              kernel_size=3,
                              mode='classification',
                              learning_rate=0.003)

        # Fit trained model
        model.fit(dataset, nb_epoch=100)

        # Eval model on train
        scores = model.evaluate(dataset, [classification_metric])
        assert scores[classification_metric.name] > 0.9

    @flaky
    @pytest.mark.torch
    def test_residual_cnn_classification(self):
        """Test that a residual CNN can overfit simple classification datasets."""
        n_samples = 10
        n_features = 3
        n_tasks = 1

        np.random.seed(123)
        X = np.random.rand(n_samples, 10, n_features)
        y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
        dataset = dc.data.NumpyDataset(X, y)

        classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        model = dc.models.CNN(n_tasks,
                              n_features,
                              dims=1,
                              dropouts=0,
                              layer_filters=[30] * 10,
                              kernel_size=3,
                              mode='classification',
                              padding='same',
                              residual=True,
                              learning_rate=0.003)

        # Fit trained model
        model.fit(dataset, nb_epoch=100)

        # Eval model on train
        scores = model.evaluate(dataset, [classification_metric])
        assert scores[classification_metric.name] > 0.9

    @flaky
    @pytest.mark.torch
    def test_cnn_regression_uncertainty(self):
        """Test computing uncertainty for a CNN regression model."""
        n_samples = 10
        n_features = 2
        n_tasks = 1
        noise = 0.1

        np.random.seed(123)
        X = np.random.randn(n_samples, 10, n_features)
        y = np.sum(X, axis=(1, 2)) + np.random.normal(scale=noise,
                                                      size=(n_samples,))
        y = np.reshape(y, (n_samples, n_tasks))
        dataset = dc.data.NumpyDataset(X, y)

        model = dc.models.CNN(n_tasks,
                              n_features,
                              dims=1,
                              dropouts=0.1,
                              kernel_size=3,
                              pool_type='average',
                              mode='regression',
                              learning_rate=0.005,
                              uncertainty=True)

        # Fit trained model
        model.fit(dataset, nb_epoch=300)

        # Predict the output and uncertainty.
        pred, std = model.predict_uncertainty(dataset)
        assert np.mean(np.abs(y - pred)) < 0.3
        assert noise < np.mean(std) < 1.0
