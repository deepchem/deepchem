import os
import deepchem as dc
import numpy as np
import pytest


def load_unlabelled_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = []
    input_file = os.path.join(current_dir, "../../data/tests/no_labels.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    return loader.create_dataset(input_file)


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


def test_transform_unlabelled():
    ul_dataset = load_unlabelled_data()
    # transforming y should raise an exception
    with pytest.raises(ValueError):
        dc.trans.transformers.Transformer(
            transform_y=True).transform(ul_dataset)

    # transforming w should raise an exception
    with pytest.raises(ValueError):
        dc.trans.transformers.Transformer(
            transform_w=True).transform(ul_dataset)

    # transforming X should be okay
    dc.trans.NormalizationTransformer(transform_X=True,
                                      dataset=ul_dataset).transform(ul_dataset)


def test_y_normalization_transformer():
    """Tests normalization transformer."""
    solubility_dataset = load_solubility_data()
    normalization_transformer = dc.trans.NormalizationTransformer(
        transform_y=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = normalization_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check that y_t has zero mean, unit std.
    assert np.isclose(y_t.mean(), 0.)
    assert np.isclose(y_t.std(), 1.)

    # Check that untransform does the right thing.
    np.testing.assert_allclose(normalization_transformer.untransform(y_t), y)


def test_X_normalization_transformer():
    """Tests normalization transformer."""
    solubility_dataset = load_solubility_data()
    normalization_transformer = dc.trans.NormalizationTransformer(
        transform_X=True, dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = normalization_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)
    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check that X_t has zero mean, unit std.
    # np.set_printoptions(threshold='nan')
    mean = X_t.mean(axis=0)
    assert np.amax(np.abs(mean - np.zeros_like(mean))) < 1e-7
    orig_std_array = X.std(axis=0)
    std_array = X_t.std(axis=0)
    # Entries with zero std are not normalized
    for orig_std, std in zip(orig_std_array, std_array):
        if not np.isclose(orig_std, 0):
            assert np.isclose(std, 1)

    # Check that untransform does the right thing.
    np.testing.assert_allclose(normalization_transformer.untransform(X_t),
                               X,
                               atol=1e-7)
