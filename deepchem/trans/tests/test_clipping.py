import deepchem as dc
import numpy as np


def test_clipping_X_transformer():
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


def test_clipping_y_transformer():
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
