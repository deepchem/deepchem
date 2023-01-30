import numpy as np
import tempfile
import deepchem as dc


def test_binary_1d():
    """Test balancing transformer on single-task dataset without explicit task dimension."""
    n_samples = 6
    n_features = 3
    np.random.seed(123)
    X = np.random.rand(n_samples, n_features)
    y = np.array([1, 1, 0, 0, 0, 0])
    w = np.ones((n_samples,))
    dataset = dc.data.NumpyDataset(X, y, w)

    duplicator = dc.trans.DuplicateBalancingTransformer(dataset=dataset)
    dataset = duplicator.transform(dataset)
    # Check that we have length 8 now with duplication
    assert len(dataset) == 8
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check shapes
    assert X_t.shape == (8, n_features)
    assert y_t.shape == (8,)
    assert w_t.shape == (8,)
    assert ids_t.shape == (8,)
    # Check that we have 4 positives and 4 negatives
    assert np.sum(y_t == 0) == 4
    assert np.sum(y_t == 1) == 4
    # Check that sum of 0s equals sum of 1s in transformed for each task
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 1]))


def test_binary_weighted_1d():
    """Test balancing transformer on a weighted single-task dataset without explicit task dimension."""
    n_samples = 6
    n_features = 3
    np.random.seed(123)
    X = np.random.rand(n_samples, n_features)
    # Note that nothing should change in this dataset since weights balance!
    y = np.array([1, 1, 0, 0, 0, 0])
    w = np.array([2, 2, 1, 1, 1, 1])
    dataset = dc.data.NumpyDataset(X, y, w)

    duplicator = dc.trans.DuplicateBalancingTransformer(dataset=dataset)
    dataset = duplicator.transform(dataset)
    # Check that still we have length 6
    assert len(dataset) == 6
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check shapes
    assert X_t.shape == (6, n_features)
    assert y_t.shape == (6,)
    assert w_t.shape == (6,)
    assert ids_t.shape == (6,)
    # Check that we have 2 positives and 4 negatives
    assert np.sum(y_t == 0) == 4
    assert np.sum(y_t == 1) == 2
    # Check that sum of 0s equals sum of 1s in transformed for each task
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 1]))


def test_binary_singletask():
    """Test duplicate balancing transformer on single-task dataset."""
    n_samples = 6
    n_features = 3
    n_tasks = 1
    np.random.seed(123)
    X = np.random.rand(n_samples, n_features)
    y = np.reshape(np.array([1, 1, 0, 0, 0, 0]), (n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w)

    duplicator = dc.trans.DuplicateBalancingTransformer(dataset=dataset)
    dataset = duplicator.transform(dataset)
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check that we have length 8 now with duplication
    assert len(dataset) == 8
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check shapes
    assert X_t.shape == (8, n_features)
    assert y_t.shape == (8,)
    assert w_t.shape == (8,)
    assert ids_t.shape == (8,)
    # Check that we have 4 positives and 4 negatives
    assert np.sum(y_t == 0) == 4
    assert np.sum(y_t == 1) == 4
    # Check that sum of 0s equals sum of 1s in transformed for each task
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 1]))


def test_multiclass_singletask():
    """Test balancing transformer on single-task dataset."""
    n_samples = 10
    n_features = 3
    X = np.random.rand(n_samples, n_features)
    # 6-1 imbalance in favor of class 0
    y = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    w = np.ones((n_samples,))
    dataset = dc.data.NumpyDataset(X, y, w)

    duplicator = dc.trans.DuplicateBalancingTransformer(dataset=dataset)
    dataset = duplicator.transform(dataset)
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)

    # Check that we have length 30 now with duplication
    assert len(dataset) == 30
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check shapes
    assert X_t.shape == (30, n_features)
    assert y_t.shape == (30,)
    assert w_t.shape == (30,)
    assert ids_t.shape == (30,)
    # Check that we have 6 of each class
    assert np.sum(y_t == 0) == 6
    assert np.sum(y_t == 1) == 6
    assert np.sum(y_t == 2) == 6
    assert np.sum(y_t == 3) == 6
    assert np.sum(y_t == 4) == 6
    # Check that sum of all class weights is equal by comparing to 0 weight
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 1]))
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 2]))
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 3]))
    assert np.isclose(np.sum(w_t[y_t == 0]), np.sum(w_t[y_t == 4]))


def test_transform_to_directory():
    """Test that output can be written to a directory."""
    n_samples = 10
    n_features = 3
    np.random.seed(123)
    X = np.random.rand(n_samples, n_features)
    # Note class imbalance. This will round to 2x duplication for 1
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    w = np.ones((n_samples,))
    dataset = dc.data.NumpyDataset(X, y, w)

    duplicator = dc.trans.DuplicateBalancingTransformer(dataset=dataset)
    with tempfile.TemporaryDirectory() as tmpdirname:
        dataset = duplicator.transform(dataset, out_dir=tmpdirname)
        balanced_dataset = dc.data.DiskDataset(tmpdirname)
        X_t, y_t, w_t, ids_t = (balanced_dataset.X, balanced_dataset.y,
                                balanced_dataset.w, balanced_dataset.ids)
        # Check that we have length 13 now with duplication
        assert len(balanced_dataset) == 13
    # Check shapes
    assert X_t.shape == (13, n_features)
    assert y_t.shape == (13,)
    assert w_t.shape == (13,)
    assert ids_t.shape == (13,)
    # Check that we have 6 positives and 7 negatives
    assert np.sum(y_t == 0) == 7
    assert np.sum(y_t == 1) == 6
