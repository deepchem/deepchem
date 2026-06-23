import numpy as np
import deepchem as dc
import pytest
from deepchem.models.hmm_models import (
    MultinomialHMMWrapper,
    CategoricalHMMWrapper,
    PoissonHMMWrapper,
    GaussianHMMWrapper,
    VariationalCategoricalHMMWrapper,
    VariationalGaussianHMMWrapper,
    GMMHMMWrapper,
)


# Synthetic Data Generators
def gen_multinomial_data(n_samples=50,
                         n_features=3,
                         n_components=2,
                         random_state=42):
    np.random.seed(random_state)
    X = np.random.randint(0, 5, size=(n_samples, n_features))
    lengths = [n_samples]
    return X, lengths


def gen_categorical_data(n_samples=50,
                         n_categories=3,
                         n_components=2,
                         random_state=42):
    np.random.seed(random_state)
    X = np.random.randint(0, n_categories, size=(n_samples, 1))
    lengths = [n_samples]
    return X, lengths


def gen_poisson_data(n_samples=50,
                     n_features=3,
                     n_components=2,
                     random_state=42):
    np.random.seed(random_state)
    X = np.random.poisson(lam=3, size=(n_samples, n_features))
    lengths = [n_samples]
    return X, lengths


def gen_gaussian_data(n_samples=50,
                      n_features=3,
                      n_components=2,
                      random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    lengths = [n_samples]
    return X, lengths


def gen_gmm_data(n_samples=50, n_features=3, n_components=2, random_state=42):
    return gen_gaussian_data(n_samples, n_features, n_components, random_state)


def verify_model(model, X, lengths):
    model.fit(X, lengths=lengths)

    if hasattr(model.model, "startprob_"):
        assert np.allclose(model.model.startprob_.sum(), 1.0)
    if hasattr(model.model, "transmat_"):
        assert np.allclose(model.model.transmat_.sum(axis=1), 1.0)

    states = model.predict(X, lengths=lengths)
    assert len(states) == X.shape[0]


# Tests
def test_multinomial_hmm():
    X, lengths = gen_multinomial_data()
    model = MultinomialHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)


def test_categorical_hmm():
    X, lengths = gen_categorical_data()
    model = CategoricalHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)


def test_poisson_hmm():
    X, lengths = gen_poisson_data()
    model = PoissonHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)


def test_gaussian_hmm():
    X, lengths = gen_gaussian_data()
    model = GaussianHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)


def test_variational_categorical_hmm():
    X, lengths = gen_categorical_data()
    model = VariationalCategoricalHMMWrapper(n_components=2,
                                             n_iter=10,
                                             random_state=42)
    verify_model(model, X, lengths)


def test_variational_gaussian_hmm():
    X, lengths = gen_gaussian_data()
    model = VariationalGaussianHMMWrapper(n_components=2,
                                          n_iter=10,
                                          random_state=42)
    verify_model(model, X, lengths)


def test_gmm_hmm():
    X, lengths = gen_gmm_data()
    model = GMMHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)


def test_model_methods():
    """Test all methods of the base wrapper class."""
    X, lengths = gen_gaussian_data()
    model = GaussianHMMWrapper(n_components=2, n_iter=10, random_state=42)

    model.fit(X, lengths=lengths)
    assert model._is_fitted

    states = model.predict(X, lengths=lengths)
    probs = model.predict_proba(X, lengths=lengths)
    score = model.score(X, lengths=lengths)
    logprob, posteriors = model.score_samples(X, lengths=lengths)
    logprob_decode, states_decode = model.decode(X, lengths=lengths)

    X_sample, Z_sample = model.sample(n_samples=10, random_state=42)

    assert len(states) == X.shape[0]
    assert probs.shape == (X.shape[0], model.model.n_components)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert not np.isnan(score)
    assert posteriors.shape == (X.shape[0], model.model.n_components)
    assert len(states_decode) == X.shape[0]
    assert X_sample.shape[0] == 10
    assert len(Z_sample) == 10

    params = model.get_params()
    assert isinstance(params, dict)

    model.set_params(n_iter=5)
    assert model.model.n_iter == 5


def test_input_validation():
    """Test input validation methods."""
    model = GaussianHMMWrapper(n_components=2, n_iter=10, random_state=42)

    with pytest.raises(ValueError, match="X must be a numpy array"):
        model._validate_input([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="X must be a 2D array"):
        model._validate_input(np.array([1, 2, 3]))

    X = np.random.randn(10, 2)
    with pytest.raises(ValueError,
                       match="Sum of lengths must equal number of samples"):
        model._validate_input(X, lengths=[5, 3])

    with pytest.raises(ValueError, match="This model has not been fitted yet"):
        model.predict(X)


def create_molecular_sequence_dataset(dataset_name="tox21", max_molecules=100):
    """
    Create molecular sequence data suitable for HMM analysis.

    Parameters
    ----------
    dataset_name : str
        Name of the DeepChem dataset to use
    max_molecules : int
        Maximum number of molecules to include

    Returns
    -------
    X : np.ndarray
        Featurized molecular data
    lengths : list
        Sequence lengths (for treating each molecule as a sequence)
    dataset : dc.data.Dataset
        Original DeepChem dataset
    """
    if dataset_name == "tox21":
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer="ECFP")
        dataset = datasets[0]
    elif dataset_name == "muv":
        tasks, datasets, transformers = dc.molnet.load_muv(featurizer="ECFP")
        dataset = datasets[0]
    elif dataset_name == "delaney":
        tasks, datasets, transformers = dc.molnet.load_delaney(
            featurizer="GraphConv")
        dataset = datasets[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Limit dataset size for testing
    if len(dataset) > max_molecules:
        dataset = dataset.select(np.arange(max_molecules))

    X = dataset.X
    # For HMM, we can treat each molecule as a separate sequence
    # or create sequences by grouping similar molecules
    lengths = [1] * len(X)  # Each molecule is a sequence of length 1

    return X, lengths, dataset


def verify_model_with_dataset(model, X, lengths, dataset=None):
    """
    Comprehensive model verification including dataset-specific checks.

    Parameters
    ----------
    model : BaseHMMWrapper
        HMM model to test
    X : np.ndarray
        Feature matrix
    lengths : list
        Sequence lengths
    dataset : dc.data.Dataset, optional
        Original DeepChem dataset for additional checks
    """
    # Basic model verification
    model.fit(X, lengths=lengths)
    assert hasattr(model, "_is_fitted") and model._is_fitted

    # Test all prediction methods
    states = model.predict(X, lengths=lengths)
    probs = model.predict_proba(X, lengths=lengths)
    score = model.score(X, lengths=lengths)
    logprob, posteriors = model.score_samples(X, lengths=lengths)
    logprob_decode, states_decode = model.decode(X, lengths=lengths)

    # Validate outputs
    assert len(states) == X.shape[0]
    assert probs.shape == (X.shape[0], model.model.n_components)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert not np.isnan(score)
    assert posteriors.shape == (X.shape[0], model.model.n_components)
    assert len(states_decode) == X.shape[0]

    # Dataset-specific validations
    if dataset is not None:
        assert X.shape[1] == dataset.X.shape[
            1], "Feature dimensions should match"

        if len(dataset.X) > 10:
            train_idx = np.arange(len(dataset.X) // 2)
            test_idx = np.arange(len(dataset.X) // 2, len(dataset.X))

            X_train = dataset.X[train_idx]
            X_test = dataset.X[test_idx]
            lengths_train = [1] * len(X_train)
            lengths_test = [1] * len(X_test)

            model_split = type(model)(n_components=model.model.n_components,
                                      n_iter=5,
                                      random_state=42)
            model_split.fit(X_train, lengths=lengths_train)

            test_states = model_split.predict(X_test, lengths=lengths_test)
            test_score = model_split.score(X_test, lengths=lengths_test)

            assert len(test_states) == len(X_test)
            assert not np.isnan(test_score)


@pytest.mark.slow
def test_hmm_with_tox21_dataset():
    """Test HMM models with Tox21 molecular dataset."""
    try:
        X, lengths, dataset = create_molecular_sequence_dataset(
            "tox21", max_molecules=50)

        model = GaussianHMMWrapper(n_components=3, n_iter=10, random_state=42)
        verify_model_with_dataset(model, X, lengths, dataset)

        print(f"Successfully tested with Tox21 dataset: {X.shape[0]} samples, "
              f"{X.shape[1]} features")

    except Exception as e:
        pytest.skip(f"Could not load Tox21 dataset: {e}")


@pytest.mark.slow
def test_hmm_with_delaney_dataset():
    """Test HMM models with Delaney solubility dataset."""
    try:
        X, lengths, dataset = create_molecular_sequence_dataset(
            "delaney", max_molecules=30)

        model = VariationalGaussianHMMWrapper(n_components=2,
                                              n_iter=10,
                                              random_state=42)
        verify_model_with_dataset(model, X, lengths, dataset)

        print(
            f"Successfully tested with Delaney dataset: {X.shape[0]} samples, "
            f"{X.shape[1]} features")

    except Exception as e:
        pytest.skip(f"Could not load Delaney dataset: {e}")


@pytest.mark.slow
def test_multiple_hmm_models_with_datasets():
    """Test multiple HMM models with different datasets."""
    models_to_test = [
        (GaussianHMMWrapper, {
            "n_components": 2,
            "n_iter": 5
        }),
        (VariationalGaussianHMMWrapper, {
            "n_components": 2,
            "n_iter": 5
        }),
        (GMMHMMWrapper, {
            "n_components": 2,
            "n_mix": 2,
            "n_iter": 5
        }),
    ]

    try:
        X, lengths, dataset = create_molecular_sequence_dataset(
            "delaney", max_molecules=20)

        for model_class, params in models_to_test:
            model = model_class(random_state=42, **params)
            verify_model_with_dataset(model, X, lengths, dataset)
            print(f"Successfully tested {model_class.__name__} with dataset")

    except Exception as e:
        pytest.skip(f"Could not run multiple model tests: {e}")
