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

from hmmlearn.vhmm import (
    VariationalCategoricalHMM,
    VariationalGaussianHMM
)


class BaseHMMWrapper:
    """Base class for all HMM wrappers providing common interface methods."""

    def fit(self, X, lengths=None):
        """
        Fit the HMM model to the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed sequence data.
        lengths : array-like of shape (n_sequences,), optional
            Lengths of individual sequences in X.

        Returns
        -------
        self : BaseHMMWrapper
            Fitted model instance.
        """
        return self.model.fit(X, lengths)

    def predict(self, X, lengths=None):
        """
        Predict the optimal hidden state sequence for the observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed sequence data.
        lengths : array-like of shape (n_sequences,), optional
            Lengths of individual sequences in X.

        Returns
        -------
        states : np.ndarray of shape (n_samples,)
            Predicted hidden states.
        """
        return self.model.predict(X, lengths)

    def predict_proba(self, X, lengths=None):
        """
        Compute posterior probabilities of each hidden state at each time step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed sequence data.
        lengths : array-like of shape (n_sequences,), optional
            Lengths of individual sequences in X.

        Returns
        -------
        posteriors : np.ndarray of shape (n_samples, n_components)
            Posterior probabilities for each state at each time step.
        """
        return self.model.predict_proba(X, lengths)

    def score(self, X, lengths=None):
        """
        Compute the log likelihood of the data under the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed sequence data.
        lengths : array-like of shape (n_sequences,), optional
            Lengths of individual sequences in X.

        Returns
        -------
        log_likelihood : float
            Log likelihood of the observed data given the model.
        """
        return self.model.score(X, lengths)

    def score_samples(self, X, lengths=None):
        """
        Compute log probabilities of observations and posterior state probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed sequence data.
        lengths : array-like of shape (n_sequences,), optional
            Lengths of individual sequences in X.

        Returns
        -------
        logprob : float
            Log probability of the observation sequence.
        posteriors : np.ndarray of shape (n_samples, n_components)
            Posterior probabilities of each hidden state at each time step.
        """
        return self.model.score_samples(X, lengths)

    def decode(self, X, lengths=None, algorithm='viterbi'):
        """
        Find the most likely state sequence corresponding to the observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed sequence data.
        lengths : array-like of shape (n_sequences,), optional
            Lengths of individual sequences in X.
        algorithm : str, default='viterbi'
            Decoding algorithm to use ('viterbi' or 'map').

        Returns
        -------
        logprob : float
            Log probability of the most likely state sequence.
        state_sequence : np.ndarray of shape (n_samples,)
            Most likely hidden state sequence.
        """
        return self.model.decode(X, lengths)


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
        """
        Initialize the GaussianHMMWrapper.

        Parameters
        ----------
        n_components : int, default=1
            Number of hidden states in the HMM.
        covariance_type : str, default='diag'
            Type of covariance parameters ('spherical', 'diag', 'full', 'tied').
        n_iter : int, default=100
            Maximum number of EM iterations to perform during fitting.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )


class MultinomialHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn.hmm.MultinomialHMM.

    Hidden Markov Model with multinomial emissions.

    Parameters
    ----------
    n_components : int, default=1
        Number of hidden states in the model.
    n_iter : int, default=100
        Maximum number of EM iterations for fitting.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    def __init__(self, n_components=1, n_iter=100, random_state=None):
        """
        Initialize the MultinomialHMMWrapper.

        Parameters
        ----------
        n_components : int, default=1
            Number of hidden states in the HMM.
        n_iter : int, default=100
            Maximum number of EM iterations to perform during fitting.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.model = MultinomialHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )


class GMMHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn.hmm.GMMHMM.

    Hidden Markov Model with Gaussian mixture emissions.

    Parameters
    ----------
    n_components : int, default=1
        Number of hidden states in the model.
    n_mix : int, default=1
        Number of Gaussian mixtures per state.
    covariance_type : str, default='diag'
        Type of covariance parameters ('spherical', 'diag', 'full', 'tied').
    n_iter : int, default=100
        Maximum number of EM iterations for fitting.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_components=1, n_mix=1, covariance_type="diag", n_iter=100, random_state=None):
        """
        Initialize the GMMHMMWrapper.

        Parameters
        ----------
        n_components : int, default=1
            Number of hidden states in the HMM.
        n_mix : int, default=1
            Number of Gaussian mixtures per state.
        covariance_type : str, default='diag'
            Type of covariance parameters ('spherical', 'diag', 'full', 'tied').
        n_iter : int, default=100
            Maximum number of EM iterations to perform during fitting.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )


class PoissonHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn.hmm.PoissonHMM.

    Hidden Markov Model with Poisson emissions.

    Parameters
    ----------
    n_components : int, default=1
        Number of hidden states in the model.
    n_iter : int, default=100
        Maximum number of EM iterations for fitting.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_components=1, n_iter=100, random_state=None):
        """
        Initialize the PoissonHMMWrapper.

        Parameters
        ----------
        n_components : int, default=1
            Number of hidden states in the HMM.
        n_iter : int, default=100
            Maximum number of EM iterations to perform during fitting.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.model = PoissonHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )


class CategoricalHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn.hmm.CategoricalHMM.

    Hidden Markov Model with categorical (discrete) emissions.

    Parameters
    ----------
    n_components : int, default=1
        Number of hidden states in the model.
    n_iter : int, default=100
        Maximum number of EM iterations for fitting.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_components=1, n_iter=100, random_state=None):
        """
        Initialize the CategoricalHMMWrapper.

        Parameters
        ----------
        n_components : int, default=1
            Number of hidden states in the HMM.
        n_iter : int, default=100
            Maximum number of EM iterations to perform during fitting.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.model = CategoricalHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )

class VariationalCategoricalHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn's VariationalCategoricalHMM.

    Hidden Markov Model with categorical (discrete) emissions trained using Variational Inference.

    Parameters
    ----------
    n_components : int, default=1
        Number of states in the model.
    n_iter : int, default=100
        Maximum number of iterations to perform during training.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    def __init__(self, n_components=1, n_iter=100, random_state=None):
        """
        Initialize the VariationalCategoricalHMMWrapper.

        Parameters
        ----------
        n_components : int, default=1
            Number of hidden states in the HMM.
        n_iter : int, default=100
            Maximum number of iterations to perform during training.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.model = VariationalCategoricalHMM(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )

    def set_fit_request(self, *, lengths='$UNCHANGED$'):
        """
        Configure how the fit method interprets sequence lengths.

        Parameters
        ----------
        lengths : bool | None | str
            Whether to use the lengths argument when fitting.

        Returns
        -------
        self : VariationalCategoricalHMMWrapper
            Self for method chaining.
        """
        self.model.set_fit_request(lengths=lengths)
        return self

class VariationalGaussianHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn's VariationalGaussianHMM.

    Hidden Markov Model with multivariate Gaussian emissions trained using Variational Inference.

    Parameters
    ----------
    n_components : int, default=1
        Number of states in the model.
    covariance_type : str, default='diag'
        Type of covariance parameters ('diag', 'full', etc.).
    n_iter : int, default=100
        Maximum number of iterations to perform during training.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    def __init__(self, n_components=1, covariance_type='diag', n_iter=100, random_state=None):
        """
        Initialize the VariationalGaussianHMMWrapper.

        Parameters
        ----------
        n_components : int, default=1
            Number of hidden states in the HMM.
        covariance_type : str, default='diag'
            Type of covariance parameters ('diag', 'full', etc.).
        n_iter : int, default=100
            Maximum number of iterations to perform during training.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.model = VariationalGaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )

    def set_fit_request(self, *, lengths='$UNCHANGED$'):
        """
        Configure how the fit method interprets sequence lengths.

        Parameters
        ----------
        lengths : bool | None | str
            Whether to use the lengths argument when fitting.

        Returns
        -------
        self : VariationalGaussianHMMWrapper
            Self for method chaining.
        """
        self.model.set_fit_request(lengths=lengths)
        return self
