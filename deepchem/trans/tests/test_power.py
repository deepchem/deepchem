import os
import deepchem as dc
import numpy as np


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


def test_power_X_transformer():
    """Test Power transformer on Gaussian normal dataset."""
    N = 10
    n_feat = 2
    powers = [1, 2, 0.5]
    X = np.random.rand(N, n_feat)
    y = np.random.normal(size=(N,))
    gaussian_dataset = dc.data.NumpyDataset(X, y)
    powers = [1, 2, 0.5]
    power_transformer = dc.trans.PowerTransformer(transform_X=True,
                                                  powers=powers)
    X, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset2 = power_transformer.transform(gaussian_dataset)
    X_t, y_t, w_t, ids_t = (gaussian_dataset2.X, gaussian_dataset2.y,
                            gaussian_dataset2.w, gaussian_dataset2.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check y is unchanged since this is an X transformer
    np.testing.assert_allclose(y.flatten(), y_t.flatten())
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check X is now holding the proper values in each column.
    np.testing.assert_allclose(X_t.shape[1], len(powers) * X.shape[1])
    np.testing.assert_allclose(X, X_t[:, :2])
    np.testing.assert_allclose(np.power(X, 2), X_t[:, 2:4])
    np.testing.assert_allclose(np.power(X, 0.5), X_t[:, 4:])


def test_power_y_transformer():
    """Test Power transformer on Gaussian normal dataset."""
    N = 10
    n_feat = 2
    powers = [1, 2, 0.5]
    X = np.random.rand(N, n_feat)
    y = np.random.rand(N)
    gaussian_dataset = dc.data.NumpyDataset(X, y)
    power_transformer = dc.trans.PowerTransformer(transform_y=True,
                                                  powers=powers)
    X, y, w, ids = (gaussian_dataset.X, gaussian_dataset.y, gaussian_dataset.w,
                    gaussian_dataset.ids)
    gaussian_dataset2 = power_transformer.transform(gaussian_dataset)
    X_t, y_t, w_t, ids_t = (gaussian_dataset2.X, gaussian_dataset2.y,
                            gaussian_dataset2.w, gaussian_dataset2.ids)

    # Check ids are unchanged.
    for id_elt, id_t_elt in zip(ids, ids_t):
        assert id_elt == id_t_elt
    # Check X is unchanged since this is an X transformer
    np.testing.assert_allclose(X, X_t)
    # Check w is unchanged since this is an X transformer
    np.testing.assert_allclose(w, w_t)
    # Check y is now holding the proper values in each column.
    np.testing.assert_allclose(y_t.shape[1], len(powers))
    np.testing.assert_allclose(y, y_t[:, :1].flatten())
    np.testing.assert_allclose(np.power(y, 2), y_t[:, 1:2].flatten())
    np.testing.assert_allclose(np.power(y, 0.5), y_t[:, 2:].flatten())

    # Check that untransform does the right thing.
    np.testing.assert_allclose(power_transformer.untransform(y_t).flatten(), y)
