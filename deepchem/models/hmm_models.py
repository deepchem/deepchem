"""
Wrappers for hmmlearn HMM models.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from hmmlearn.hmm import (GaussianHMM, MultinomialHMM, GMMHMM, PoissonHMM,
                          CategoricalHMM)

from hmmlearn.vhmm import (VariationalCategoricalHMM, VariationalGaussianHMM)


class BaseHMMWrapper:
    """Base class for all HMM wrappers providing common interface methods."""

    def __init__(self):
        self.model = None
        self._is_fitted = False

    def fit(self,
            X: np.ndarray,
            lengths: Optional[np.ndarray] = None) -> 'BaseHMMWrapper':
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
        self._validate_input(X, lengths)
        self.model.fit(X, lengths)
        self._is_fitted = True
        return self

    def predict(self,
                X: np.ndarray,
                lengths: Optional[np.ndarray] = None) -> np.ndarray:
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
        self._check_fitted()
        self._validate_input(X, lengths)
        return self.model.predict(X, lengths)

    def predict_proba(self,
                      X: np.ndarray,
                      lengths: Optional[np.ndarray] = None) -> np.ndarray:
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
        self._check_fitted()
        self._validate_input(X, lengths)
        return self.model.predict_proba(X, lengths)

    def score(self,
              X: np.ndarray,
              lengths: Optional[np.ndarray] = None) -> float:
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
        self._check_fitted()
        self._validate_input(X, lengths)
        return self.model.score(X, lengths)

    def score_samples(
            self,
            X: np.ndarray,
            lengths: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
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
        self._check_fitted()
        self._validate_input(X, lengths)
        return self.model.score_samples(X, lengths)

    def decode(self,
               X: np.ndarray,
               lengths: Optional[np.ndarray] = None,
               algorithm: str = 'viterbi') -> Tuple[float, np.ndarray]:
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
        self._check_fitted()
        self._validate_input(X, lengths)
        return self.model.decode(X, lengths, algorithm)

    def sample(
            self,
            n_samples: int = 1,
            random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Generated observations.
        Z : np.ndarray of shape (n_samples,)
            Generated hidden states.
        """
        self._check_fitted()
        return self.model.sample(n_samples, random_state=random_state)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        if self.model is not None:
            return self.model.get_params(deep)
        return {}

    def set_params(self, **params) -> 'BaseHMMWrapper':
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : BaseHMMWrapper
            Estimator instance.
        """
        if self.model is not None:
            self.model.set_params(**params)
        return self

    def _validate_input(self,
                        X: np.ndarray,
                        lengths: Optional[np.ndarray] = None):
        """Validate input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if lengths is not None:
            lengths = np.asarray(lengths)
            if np.sum(lengths) != X.shape[0]:
                raise ValueError(
                    "Sum of lengths must equal number of samples in X")

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise ValueError(
                "This model has not been fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )


class GaussianHMMWrapper(BaseHMMWrapper):
    """
    Wrapper for hmmlearn.hmm.GaussianHMM.

    Parameters
    ----------
    n_components : int, default=1
        Number of hidden states in the HMM.
    covariance_type : str, default='diag'
        Type of covariance parameters ('spherical', 'diag', 'full', 'tied').
    n_iter : int, default=100
        Maximum number of EM iterations to perform during fitting.
    tol : float, default=1e-2
        Convergence threshold for EM algorithm.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_components: int = 1,
                 covariance_type: str = "diag",
                 n_iter: int = 100,
                 tol: float = 1e-2,
                 random_state: Optional[int] = None):
        super().__init__()
        self.model = GaussianHMM(n_components=n_components,
                                 covariance_type=covariance_type,
                                 n_iter=n_iter,
                                 tol=tol,
                                 random_state=random_state)


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
    tol : float, default=1e-2
        Convergence threshold for EM algorithm.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_components: int = 1,
                 n_iter: int = 100,
                 tol: float = 1e-2,
                 random_state: Optional[int] = None):
        super().__init__()
        self.model = MultinomialHMM(n_components=n_components,
                                    n_iter=n_iter,
                                    tol=tol,
                                    random_state=random_state)


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
    tol : float, default=1e-2
        Convergence threshold for EM algorithm.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_components: int = 1,
                 n_mix: int = 1,
                 covariance_type: str = "diag",
                 n_iter: int = 100,
                 tol: float = 1e-2,
                 random_state: Optional[int] = None):
        super().__init__()
        self.model = GMMHMM(n_components=n_components,
                            n_mix=n_mix,
                            covariance_type=covariance_type,
                            n_iter=n_iter,
                            tol=tol,
                            random_state=random_state)


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
    tol : float, default=1e-2
        Convergence threshold for EM algorithm.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_components: int = 1,
                 n_iter: int = 100,
                 tol: float = 1e-2,
                 random_state: Optional[int] = None):
        super().__init__()
        self.model = PoissonHMM(n_components=n_components,
                                n_iter=n_iter,
                                tol=tol,
                                random_state=random_state)


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
    tol : float, default=1e-2
        Convergence threshold for EM algorithm.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_components: int = 1,
                 n_iter: int = 100,
                 tol: float = 1e-2,
                 random_state: Optional[int] = None):
        super().__init__()
        self.model = CategoricalHMM(n_components=n_components,
                                    n_iter=n_iter,
                                    tol=tol,
                                    random_state=random_state)


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
    tol : float, default=1e-3
        Convergence threshold.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_components: int = 1,
                 n_iter: int = 100,
                 tol: float = 1e-3,
                 random_state: Optional[int] = None):
        super().__init__()
        self.model = VariationalCategoricalHMM(n_components=n_components,
                                               n_iter=n_iter,
                                               tol=tol,
                                               random_state=random_state)

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
    tol : float, default=1e-3
        Convergence threshold.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_components: int = 1,
                 covariance_type: str = 'diag',
                 n_iter: int = 100,
                 tol: float = 1e-3,
                 random_state: Optional[int] = None):
        super().__init__()
        self.model = VariationalGaussianHMM(n_components=n_components,
                                            covariance_type=covariance_type,
                                            n_iter=n_iter,
                                            tol=tol,
                                            random_state=random_state)

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
