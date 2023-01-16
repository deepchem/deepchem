import os
import numpy as np

import deepchem as dc


def load_gaussian_cdf_data():
    """Load example with numbers sampled from Gaussian normal distribution.
    Each feature and task is a column of values that is sampled
    from a normal distribution of mean 0, stdev 1."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features = ["feat0", "feat1"]
    featurizer = dc.feat.UserDefinedFeaturizer(features)
    tasks = ["task0", "task1"]
    input_file = os.path.join(current_dir, "assets/gaussian_cdf_example.csv")
    loader = dc.data.UserCSVLoader(tasks=tasks,
                                   featurizer=featurizer,
                                   id_field="id")
    return loader.create_dataset(input_file)


def test_cdf_X_transformer():
    """Test CDF transformer on Gaussian normal dataset."""
    target = np.array(np.transpose(np.linspace(0., 1., 1001)))
    target = np.transpose(np.array(np.append([target], [target], axis=0)))
    gaussian_dataset = load_gaussian_cdf_data()
    bins = 1001
    cdf_transformer = dc.trans.CDFTransformer(transform_X=True,
                                              dataset=gaussian_dataset,
                                              bins=bins)
    _, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset = cdf_transformer.transform(gaussian_dataset)
    X_t, y_t, w_t, ids_t = (gaussian_dataset.X, gaussian_dataset.y,
                            gaussian_dataset.w, gaussian_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check y is unchanged since this is an X transformer
    np.testing.assert_allclose(y, y_t)
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check X is now holding the proper values when sorted.
    sorted = np.sort(X_t, axis=0)
    np.testing.assert_allclose(sorted, target)


def test_cdf_1d_y_transformer():
    """Test on a synthetic dataset we sample with 1d y."""
    N = 10
    n_feat = 5
    n_bins = 100
    X = np.random.normal(size=(N, n_feat))
    y = np.random.normal(size=(N,))
    dataset = dc.data.NumpyDataset(X, y)
    cdftrans = dc.trans.CDFTransformer(transform_y=True,
                                       dataset=dataset,
                                       bins=n_bins)
    dataset = cdftrans.transform(dataset)


def test_cdf_y_transformer():
    """Test CDF transformer on Gaussian normal dataset."""
    target = np.array(np.transpose(np.linspace(0., 1., 1001)))
    target = np.transpose(np.array(np.append([target], [target], axis=0)))
    gaussian_dataset = load_gaussian_cdf_data()
    bins = 1001
    cdf_transformer = dc.trans.CDFTransformer(transform_y=True,
                                              dataset=gaussian_dataset,
                                              bins=bins)
    X, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset = cdf_transformer.transform(gaussian_dataset, bins=bins)
    X_t, y_t, w_t, ids_t = (gaussian_dataset.X, gaussian_dataset.y,
                            gaussian_dataset.w, gaussian_dataset.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is an y transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is an y transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now holding the proper values when sorted.
    sorted = np.sort(y_t, axis=0)
    np.testing.assert_allclose(sorted, target)

    # Check that untransform does the right thing.
    y_restored = cdf_transformer.untransform(y_t)
    assert np.max(y_restored - y) < 1e-5
