import numpy as np
import pytest
from deepchem.models.hmm_models import (
    MultinomialHMMWrapper,
    CategoricalHMMWrapper,
    PoissonHMMWrapper,
    GaussianHMMWrapper,
    VariationalCategoricalHMMWrapper,
    VariationalGaussianHMMWrapper,
    GMMHMMWrapper
)

# Synthetic Data Generators
def gen_multinomial_data(n_samples=50, n_features=3, n_components=2, random_state=42):
    np.random.seed(random_state)
    X = np.random.randint(0, 5, size=(n_samples, n_features))
    lengths = [n_samples]
    return X, lengths

def gen_categorical_data(n_samples=50, n_categories=3, n_components=2, random_state=42):
    np.random.seed(random_state)
    X = np.random.randint(0, n_categories, size=(n_samples, 1))
    lengths = [n_samples]
    return X, lengths

def gen_poisson_data(n_samples=50, n_features=3, n_components=2, random_state=42):
    np.random.seed(random_state)
    X = np.random.poisson(lam=3, size=(n_samples, n_features))
    lengths = [n_samples]
    return X, lengths

def gen_gaussian_data(n_samples=50, n_features=3, n_components=2, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    lengths = [n_samples]
    return X, lengths

def gen_gmm_data(n_samples=50, n_features=3, n_components=2, random_state=42):
    return gen_gaussian_data(n_samples, n_features, n_components, random_state)

# Generic Verification Function
def verify_model(model, X, lengths):
    model.fit(X, lengths=lengths)
    
    # Check start probabilities sum to 1
    if hasattr(model.model, "startprob_"):
        assert np.allclose(model.model.startprob_.sum(), 1.0)
    # Check transition matrix rows sum to 1
    if hasattr(model.model, "transmat_"):
        assert np.allclose(model.model.transmat_.sum(axis=1), 1.0)
    
    # Predict and check length
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
    model = VariationalCategoricalHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)

def test_variational_gaussian_hmm():
    X, lengths = gen_gaussian_data()
    model = VariationalGaussianHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)

def test_gmm_hmm():
    X, lengths = gen_gmm_data()
    model = GMMHMMWrapper(n_components=2, n_iter=10, random_state=42)
    verify_model(model, X, lengths)
