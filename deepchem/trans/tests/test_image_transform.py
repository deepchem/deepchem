import numpy as np
import deepchem as dc


def test_x_image_transform():
    """Test ImageTransformer on only X array"""
    n_samples = 10
    width = 35
    height = 67
    channels = 4
    new_size = (10, 15)

    X = np.random.randn(n_samples, width, height, channels)
    y = np.random.randn(n_samples, 1)
    w = np.random.randn(n_samples, 1)
    ids = np.random.randn(n_samples, 1)

    dataset = dc.data.ImageDataset(X, y, w, ids)
    img_transformer = dc.trans.ImageTransformer(size=new_size,
                                                transform_X=True,
                                                transform_y=False)
    dataset = dataset.transform(img_transformer)

    assert dataset.X.shape == (n_samples, new_size[1], new_size[0], channels)
    assert dataset.y.shape == (n_samples, 1)
    assert dataset.w.shape == (n_samples, 1)
    assert dataset.ids.shape == (n_samples, 1)


def test_y_image_transform():
    """Test ImageTransformer on only y array"""
    n_samples = 10
    width = 35
    height = 67
    channels = 4
    new_size = (10, 15)

    X = np.random.randn(n_samples, width, height, channels)
    y = np.random.randn(n_samples, width, height, channels)
    w = np.random.randn(n_samples, 1)
    ids = np.random.randn(n_samples, 1)

    dataset = dc.data.ImageDataset(X, y, w, ids)
    img_transformer = dc.trans.ImageTransformer(size=new_size,
                                                transform_X=False,
                                                transform_y=True)
    dataset = dataset.transform(img_transformer)

    assert dataset.X.shape == (n_samples, width, height, channels)
    assert dataset.y.shape == (n_samples, new_size[1], new_size[0], channels)
    assert dataset.w.shape == (n_samples, 1)
    assert dataset.ids.shape == (n_samples, 1)


def test_xy_image_transform():
    """Test ImageTransformer on both X and y arrays"""
    n_samples = 10
    width = 35
    height = 67
    channels = 4
    new_size = (10, 15)

    X = np.random.randn(n_samples, width, height, channels)
    y = np.random.randn(n_samples, width, height, channels)
    w = np.random.randn(n_samples, 1)
    ids = np.random.randn(n_samples, 1)

    dataset = dc.data.ImageDataset(X, y, w, ids)
    img_transformer = dc.trans.ImageTransformer(size=new_size,
                                                transform_X=True,
                                                transform_y=True)
    dataset = dataset.transform(img_transformer)

    assert dataset.X.shape == (n_samples, new_size[1], new_size[0], channels)
    assert dataset.y.shape == (n_samples, new_size[1], new_size[0], channels)
    assert dataset.w.shape == (n_samples, 1)
    assert dataset.ids.shape == (n_samples, 1)


def test_image_transform_array():
    """Test ImageTransformer on random array"""
    n_samples = 10
    width = 35
    height = 67
    channels = 4
    new_size = (10, 15)

    X = np.random.randn(n_samples, width, height, channels)
    y = np.random.randn(n_samples, width, height, channels)
    w = np.random.randn(n_samples, 1)
    ids = np.random.randn(n_samples, 1)

    img_transform = dc.trans.ImageTransformer(size=new_size,
                                              transform_X=True,
                                              transform_y=True)
    data = img_transform.transform_array(X, y, w, ids)

    assert data[0].shape == (n_samples, new_size[1], new_size[0], channels)
    assert data[1].shape == (n_samples, new_size[1], new_size[0], channels)
    assert data[2].shape == (n_samples, 1)
    assert data[3].shape == (n_samples, 1)
