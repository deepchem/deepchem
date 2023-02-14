import unittest
import tempfile

import pytest
import deepchem as dc
import numpy as np

try:
    import tensorflow as tf  # noqa: F401
    has_tensorflow = True
except:
    has_tensorflow = False


class TestScScoreModel(unittest.TestCase):

    @pytest.mark.tensorflow
    def test_overfit_scscore(self):
        """Test fitting to a small dataset"""
        n_samples = 10
        n_features = 3
        n_tasks = 1

        # Create a dataset and an input function for processing it.

        X = np.random.rand(n_samples, 2, n_features)
        y = np.random.randint(2, size=(n_samples, n_tasks))
        dataset = dc.data.NumpyDataset(X, y)

        model = dc.models.ScScoreModel(n_features, dropouts=0)

        model.fit(dataset, nb_epoch=100)
        pred = model.predict(dataset)
        assert np.array_equal(y, pred[0] > pred[1])


@pytest.mark.tensorflow
def test_scscore_reload():
    """Test reloading of ScScoreModel"""
    n_samples = 10
    n_features = 3
    n_tasks = 1

    # Create a dataset and an input function for processing it.

    X = np.random.rand(n_samples, 2, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y)

    model_dir = tempfile.mkdtemp()
    model = dc.models.ScScoreModel(n_features, dropouts=0, model_dir=model_dir)
    model.fit(dataset, nb_epoch=100)
    pred = model.predict(dataset)
    assert np.array_equal(y, pred[0] > pred[1])

    reloaded_model = dc.models.ScScoreModel(n_features,
                                            dropouts=0,
                                            model_dir=model_dir)
    reloaded_model.restore()
    reloaded_pred = reloaded_model.predict(dataset)
    assert len(pred) == len(reloaded_pred)
    for p, r in zip(pred, reloaded_pred):
        assert np.all(p == r)
