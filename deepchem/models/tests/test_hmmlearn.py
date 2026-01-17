"""Tests for Hidden Markov Model wrapper."""
import numpy as np
import pytest
import deepchem as dc
pytest.importorskip("hmmlearn")
from hmmlearn.hmm import GaussianHMM  # noqa: E402
from deepchem.models.hidden_markov_model.markov_models import HiddenMarkovModel  # noqa: E402


def test_hmm_fit_predict():
    """Test that HMM model can fit and predict."""
    np.random.seed(123)
    X = np.random.randn(100, 2)
    lengths = [50, 50]
    dataset = dc.data.NumpyDataset(X=X)
    hmm = GaussianHMM(n_components=3, n_iter=100, random_state=123)
    model = HiddenMarkovModel(hmm)
    model.fit(dataset, lengths=lengths)
    predictions = model.predict(dataset=dataset, lengths=lengths)
    assert predictions.shape[0] == X.shape[0]
    assert len(np.unique(predictions)) <= 3


def test_hmm_save_reload():
    """Test that HMM model can be saved and reloaded."""
    np.random.seed(123)
    X = np.random.randn(50, 2)
    lengths = [50]
    dataset = dc.data.NumpyDataset(X=X)
    hmm = GaussianHMM(n_components=2, n_iter=50, random_state=123)
    model = HiddenMarkovModel(hmm)
    model.fit(dataset, lengths=lengths)
    pred_before = model.predict(dataset=dataset, lengths=lengths)
    model.save()
    model.reload()
    pred_after = model.predict(dataset=dataset, lengths=lengths)

    assert np.array_equal(pred_before, pred_after)


def test_hmm_with_transformers():
    """Test HMM with data transformers."""
    np.random.seed(123)
    X = np.random.randn(100, 2) * 100
    lengths = [100]

    dataset = dc.data.NumpyDataset(X=X)
    transformer = dc.trans.NormalizationTransformer(
        transform_X=True,
        dataset=dataset
    )
    dataset = transformer.transform(dataset)

    hmm = GaussianHMM(n_components=3, n_iter=100, random_state=123)
    model = HiddenMarkovModel(hmm)
    model.fit(dataset, lengths=lengths)

    predictions = model.predict(
        dataset=dataset,
        transformers=[transformer],
        lengths=lengths
    )

    assert predictions.shape[0] == X.shape[0]


@pytest.mark.parametrize("n_components", [2, 3, 5])
def test_hmm_different_components(n_components):
    """Test HMM with different number of hidden states."""
    np.random.seed(123)
    X = np.random.randn(100, 2)
    lengths = [100]
    dataset = dc.data.NumpyDataset(X=X)
    hmm = GaussianHMM(n_components=n_components, n_iter=50, random_state=123)
    model = HiddenMarkovModel(hmm)
    model.fit(dataset, lengths=lengths)
    predictions = model.predict(dataset=dataset, lengths=lengths)
    assert len(np.unique(predictions)) <= n_components
