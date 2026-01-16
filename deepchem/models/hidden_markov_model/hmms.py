

import pandas as pd
import inspect
import logging
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator

from deepchem.models import Model
from deepchem.data import Dataset
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.data_utils import load_from_disk, save_to_disk
from deepchem.utils.typing import OneOrMany

logger = logging.getLogger(__name__)


class HiddenMarkovModel(Model): 
    """
    DeepChem-compatible wrapper for scikit-learn–style Hidden Markov Models.

    This class adapts a Hidden Markov Model (HMM) that follows the scikit-learn
    estimator interface (e.g., from `hmmlearn`) so that it can be used inside
    the DeepChem `Model` API. It enables training, prediction, saving, and
    reloading of HMMs using DeepChem datasets and utilities.
    """

    def __init__(self, 
                 model: BaseEstimator,
                 model_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the HiddenMarkovModel wrapper.

        Parameters
        ----------
        model : BaseEstimator
            A scikit-learn–compatible Hidden Markov Model instance
            (for example, `hmmlearn.hmm.GaussianHMM`).
        model_dir : Optional[str], default=None
            Directory where the model will be saved and loaded from.
        **kwargs
            Additional keyword arguments passed to the DeepChem `Model` base class.
        """
        super(HiddenMarkovModel, self).__init__(model, model_dir, **kwargs)
        self.model = model

    def fit(self, dataset: Dataset, lengths=None) -> None:
        """
        Fit the Hidden Markov Model on a DeepChem dataset.

        This method extracts the feature matrix from the dataset and trains
        the underlying HMM using sequence length information.

        Parameters
        ----------
        dataset : Dataset
            A DeepChem `Dataset` containing the input sequences in `dataset.X`.
        lengths : Optional[list], default=None
            A list specifying the lengths of individual sequences in `dataset.X`.
            If not provided, all sequences are assumed to be of length 1.
        """
        X = dataset.X
        if lengths is None:
            lengths = [1] * len(X.shape)

        self.model.fit(X, lengths)
        return

    def predict(self,
                dataset: Dataset,
                transformers: List[Transformer] = [],
                lengths=None) -> OneOrMany[np.ndarray]:
        """
        Generate predictions for a DeepChem dataset.

        The method runs inference using the underlying HMM and then applies
        inverse transformations (if any) to return predictions in the original
        data space.

        Parameters
        ----------
        dataset : Dataset
            A DeepChem `Dataset` containing input sequences in `dataset.X`.
        transformers : List[Transformer], default=[]
            List of DeepChem transformers used during preprocessing. These are
            reversed before returning predictions.
        lengths : Optional[list], default=None
            A list specifying the lengths of individual sequences. If not provided,
            all sequences are assumed to be of length 1.

        Returns
        -------
        OneOrMany[np.ndarray]
            Model predictions after undoing the applied transformers.
        """
        X = dataset.X
        if lengths is None:
            lengths = [1] * len(X.shape)

        y_pred = self.model.predict(X, lengths)
        return undo_transforms(y_pred, transformers)

    def predict_on_batch(self, X: Dataset, lengths=None):
        """
        Generate predictions for a single batch of data.

        This method attempts to return probabilistic predictions if the
        underlying model supports `predict_proba`. Otherwise, it falls back
        to standard predictions.

        Parameters
        ----------
        X : Dataset
            A DeepChem `Dataset` containing a batch of input data in `X.X`.
        lengths : Optional[list], default=None
            A list specifying the sequence lengths for the batch.

        Returns
        -------
        np.ndarray
            Predicted labels or probabilities for the batch.
        """
        X = X.X
        try:
            return self.model.predict_proba(X, lengths)
        except AttributeError:
            return self.model.predict(X, lengths)

    def save(self):
        """
        Save the underlying HMM model to disk.

        The model is serialized using DeepChem's disk utilities and stored
        in the directory specified by `model_dir`.
        """
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """
        Reload the HMM model from disk.

        This method restores the previously saved model from `model_dir`
        and assigns it back to `self.model`.
        """
        self.model = load_from_disk(self.get_model_filename(self.model_dir))