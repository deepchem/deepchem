import os
import numpy as np
import deepchem as dc


def load_solubility_data():
    """Loads solubility dataset"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    input_file = os.path.join(current_dir,
                              "../../models/tests/assets/example.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    return loader.create_dataset(input_file)


def test_y_minmax_transformer():
    """Tests MinMax transformer."""
    solubility_dataset = load_solubility_data()
    minmax_transformer = dc.trans.MinMaxTransformer(transform_y=True,
                                                    dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = minmax_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged before and after transformation
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt

    # Check X is unchanged since transform_y is true
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since transform_y is true
    np.testing.assert_allclose(w, w_t)

    # Check minimum and maximum values of transformed y are 0 and 1
    np.testing.assert_allclose(y_t.min(), 0.)
    np.testing.assert_allclose(y_t.max(), 1.)

    # Check untransform works correctly
    y_restored = minmax_transformer.untransform(y_t)
    assert np.max(y_restored - y) < 1e-5


def test_y_minmax_random():
    """Test on random example"""
    n_samples = 100
    n_features = 10
    n_tasks = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_tasks)
    dataset = dc.data.NumpyDataset(X, y)

    minmax_transformer = dc.trans.MinMaxTransformer(transform_y=True,
                                                    dataset=dataset)
    w, ids = dataset.w, dataset.ids

    dataset = minmax_transformer.transform(dataset)
    X_t, y_t, w_t, ids_t = (dataset.X, dataset.y, dataset.w, dataset.ids)
    # Check ids are unchanged before and after transformation
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt

    # Check X is unchanged since transform_y is true
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since transform_y is true
    np.testing.assert_allclose(w, w_t)

    # Check minimum and maximum values of transformed y are 0 and 1
    np.testing.assert_allclose(y_t.min(), 0.)
    np.testing.assert_allclose(y_t.max(), 1.)

    # Test if dimensionality expansion is handled correctly by untransform
    y_t = np.expand_dims(y_t, axis=-1)
    y_restored = minmax_transformer.untransform(y_t)
    assert y_restored.shape == y.shape + (1,)
    np.testing.assert_allclose(np.squeeze(y_restored, axis=-1), y)


def test_X_minmax_transformer():
    solubility_dataset = load_solubility_data()
    minmax_transformer = dc.trans.MinMaxTransformer(transform_X=True,
                                                    dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = minmax_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged before and after transformation
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt

    # Check X is unchanged since transform_y is true
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since transform_y is true
    np.testing.assert_allclose(w, w_t)

    # Check minimum and maximum values of transformed y are 0 and 1
    np.testing.assert_allclose(X_t.min(), 0.)
    np.testing.assert_allclose(X_t.max(), 1.)

    # Check untransform works correctly
    np.testing.assert_allclose(minmax_transformer.untransform(X_t), X)


def test_minmax_untransform_shapes():
    """Test untransform preserves shapes across batch_size == 1 and N > 1."""
    np.random.seed(123)
    n_samples = 11
    n_features = 5
    n_tasks = 2
    train_dataset = dc.data.NumpyDataset(np.random.rand(n_samples, n_features),
                                         np.random.rand(n_samples, n_tasks))

    transformer = dc.trans.MinMaxTransformer(transform_y=True,
                                             dataset=train_dataset)

    y_min_3d = transformer.y_min.reshape(1, n_tasks, 1)
    y_max_3d = transformer.y_max.reshape(1, n_tasks, 1)

    # shape (1, n_tasks, 1) round-trips unchanged
    y_raw = np.random.rand(1, n_tasks, 1)
    y_t = (y_raw - y_min_3d) / (y_max_3d - y_min_3d)
    out_1 = transformer.untransform(y_t)
    assert out_1.shape == y_raw.shape
    np.testing.assert_allclose(out_1, y_raw)

    # shape (N, n_tasks) with N > 1 still behaves exactly as before
    batch_n_2d = np.random.rand(n_samples, n_tasks)
    y_t_2d = (batch_n_2d - transformer.y_min) / (transformer.y_max -
                                                 transformer.y_min)
    out_n_2d = transformer.untransform(y_t_2d)
    assert out_n_2d.shape == batch_n_2d.shape
    np.testing.assert_allclose(out_n_2d, batch_n_2d)

    # n_tasks == 1 case with shape (1, 1)
    n_tasks_1 = 1
    train_dataset_1 = dc.data.NumpyDataset(
        np.random.rand(n_samples, n_features),
        np.random.rand(n_samples, n_tasks_1))
    transformer_1 = dc.trans.MinMaxTransformer(transform_y=True,
                                               dataset=train_dataset_1)

    raw_1_1 = np.random.rand(1, 1)
    t_1_1 = (raw_1_1 - transformer_1.y_min) / (transformer_1.y_max -
                                               transformer_1.y_min)
    out_1_1 = transformer_1.untransform(t_1_1)
    assert out_1_1.shape == (1, 1)
    np.testing.assert_allclose(out_1_1, raw_1_1)
