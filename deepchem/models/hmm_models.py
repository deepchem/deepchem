"""
Wrappers for hmmlearn HMM models.
"""

from hmmlearn.hmm import (
    GaussianHMM,
    MultinomialHMM,
    GMMHMM,
    PoissonHMM,
    CategoricalHMM
)


class BaseHMMWrapper:
    """Base class for all HMM wrappers."""

    def fit(self, X, lengths=None):
        """Fit model to data X (numpy array)."""
        return self.model.fit(X, lengths)

    def predict(self, X, lengths=None):
        """Predict the optimal state sequence for X."""
        return self.model.predict(X, lengths)

    def score(self, X, lengths=None):
        """Compute the log likelihood of the data under the model."""
        return self.model.score(X, lengths)


class GaussianHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn.hmm.GaussianHMM.

    Parameters
    ----------
    n_components : int
        Number of states.
    covariance_type : str
        Type of covariance parameters ('spherical', 'diag', 'full', 'tied').
    n_iter : int
        Maximum number of EM iterations.
    """

    def __init__(self, n_components=1, covariance_type="diag", n_iter=100, random_state=None):
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )


class MultinomialHMMWrapper(BaseHMMWrapper):
    """Wrapper for hmmlearn.hmm.MultinomialHMM."""

    def __init__(self, n_components=1, n_iter=100, random_state=None):
        self.model = MultinomialHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )


class GMMHMMWrapper(BaseHMMWrapper):
    """Wrapper for hmmlearn.hmm.GMMHMM."""

    def __init__(self, n_components=1, n_mix=1, covariance_type="diag", n_iter=100, random_state=None):
        self.model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )


class PoissonHMMWrapper(BaseHMMWrapper):
    """Wrapper for hmmlearn.hmm.PoissonHMM."""

    def __init__(self, n_components=1, n_iter=100, random_state=None):
        self.model = PoissonHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )


class CategoricalHMMWrapper(BaseHMMWrapper):
    """Wrapper for hmmlearn.hmm.CategoricalHMM."""

    def __init__(self, n_components=1, n_iter=100, random_state=None):
        self.model = CategoricalHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )
