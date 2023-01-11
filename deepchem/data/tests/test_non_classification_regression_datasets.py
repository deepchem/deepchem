import deepchem as dc
import numpy as np


def test_disk_generative_dataset():
    """Test for a hypothetical generative dataset."""
    X = np.random.rand(100, 10, 10)
    y = np.random.rand(100, 10, 10)
    dataset = dc.data.DiskDataset.from_numpy(X, y)
    assert (dataset.X == X).all()
    assert (dataset.y == y).all()


def test_numpy_generative_dataset():
    """Test for a hypothetical generative dataset."""
    X = np.random.rand(100, 10, 10)
    y = np.random.rand(100, 10, 10)
    dataset = dc.data.NumpyDataset(X, y)
    assert (dataset.X == X).all()
    assert (dataset.y == y).all()
