import os
import deepchem as dc
import pandas as pd
import numpy as np


def test_log_trans_1D():
    """Test in 1D case without explicit task variable."""
    X = np.random.rand(10, 10)
    y = np.random.rand(10)
    dataset = dc.data.NumpyDataset(X, y)
    trans = dc.trans.LogTransformer(transform_y=True)
    log_dataset = trans.transform(dataset)
    assert np.isclose(np.log(y + 1), log_dataset.y).all()
    untrans_y = trans.untransform(log_dataset.y)
    assert np.isclose(untrans_y, y).all()


def load_feat_multitask_data():
    """Load example with numerical features, tasks."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features = ["feat0", "feat1", "feat2", "feat3", "feat4", "feat5"]
    featurizer = dc.feat.UserDefinedFeaturizer(features)
    tasks = ["task0", "task1", "task2", "task3", "task4", "task5"]
    input_file = os.path.join(current_dir, "assets/feat_multitask_example.csv")
    loader = dc.data.UserCSVLoader(tasks=tasks,
                                   featurizer=featurizer,
                                   id_field="id")
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


def test_y_log_transformer():
    """Tests logarithmic data transformer."""
    solubility_dataset = load_solubility_data()
    log_transformer = dc.trans.LogTransformer(transform_y=True,
                                              dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = log_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(y_t, np.log(y + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(y_t), y)


def test_X_log_transformer():
    """Tests logarithmic data transformer."""
    solubility_dataset = load_solubility_data()
    log_transformer = dc.trans.LogTransformer(transform_X=True,
                                              dataset=solubility_dataset)
    X, y, w, ids = (solubility_dataset.X, solubility_dataset.y,
                    solubility_dataset.w, solubility_dataset.ids)
    solubility_dataset = log_transformer.transform(solubility_dataset)
    X_t, y_t, w_t, ids_t = (solubility_dataset.X, solubility_dataset.y,
                            solubility_dataset.w, solubility_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(X_t, np.log(X + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(X_t), X)


def test_y_log_transformer_select():
    """Tests logarithmic data transformer with selection."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    multitask_dataset = load_feat_multitask_data()
    dfe = pd.read_csv(
        os.path.join(current_dir, "assets/feat_multitask_example.csv"))
    tid = []
    tasklist = ["task0", "task3", "task4", "task5"]
    first_task = "task0"
    for task in tasklist:
        tiid = dfe.columns.get_loc(task) - dfe.columns.get_loc(first_task)
        tid = np.concatenate((tid, np.array([tiid])))
    tasks = tid.astype(int)
    log_transformer = dc.trans.LogTransformer(transform_y=True,
                                              tasks=tasks,
                                              dataset=multitask_dataset)
    X, y, w, ids = (multitask_dataset.X, multitask_dataset.y,
                    multitask_dataset.w, multitask_dataset.ids)
    multitask_dataset = log_transformer.transform(multitask_dataset)
    X_t, y_t, w_t, ids_t = (multitask_dataset.X, multitask_dataset.y,
                            multitask_dataset.w, multitask_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is a y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(y_t[:, tasks], np.log(y[:, tasks] + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(y_t), y)


def test_X_log_transformer_select():
    # Tests logarithmic data transformer with selection.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    multitask_dataset = load_feat_multitask_data()
    dfe = pd.read_csv(
        os.path.join(current_dir, "assets/feat_multitask_example.csv"))
    fid = []
    featurelist = ["feat0", "feat1", "feat2", "feat3", "feat5"]
    first_feature = "feat0"
    for feature in featurelist:
        fiid = dfe.columns.get_loc(feature) - dfe.columns.get_loc(first_feature)
        fid = np.concatenate((fid, np.array([fiid])))
    features = fid.astype(int)
    log_transformer = dc.trans.LogTransformer(transform_X=True,
                                              features=features,
                                              dataset=multitask_dataset)
    X, y, w, ids = (multitask_dataset.X, multitask_dataset.y,
                    multitask_dataset.w, multitask_dataset.ids)
    multitask_dataset = log_transformer.transform(multitask_dataset)
    X_t, y_t, w_t, ids_t = (multitask_dataset.X, multitask_dataset.y,
                            multitask_dataset.w, multitask_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check y is unchanged since this is a X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is a y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now a logarithmic version of itself
    np.testing.assert_allclose(X_t[:, features], np.log(X[:, features] + 1))

    # Check that untransform does the right thing.
    np.testing.assert_allclose(log_transformer.untransform(X_t), X)
