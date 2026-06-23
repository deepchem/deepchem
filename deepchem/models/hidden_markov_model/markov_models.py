import logging
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator

from deepchem.models import Model
from deepchem.data import Dataset
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.data_utils import load_from_disk, save_to_disk
from deepchem.utils.typing import OneOrMany
import numpy as np

logger = logging.getLogger(__name__)


class HiddenMarkovModel(Model):
    """
    Docstring for HiddenMarkovModel
    """

    def __init__(self, model, model_dir: Optional[str] = None, **kwargs):
        super(HiddenMarkovModel, self).__init__(model, model_dir, **kwargs)
        self.model = model

    def fit(self, dataset: Dataset, lengths = None) -> None:
        """
        Docstring for fit
        
        :param self: Description
        :param dataset: Description
        :type dataset: Dataset
        :param lengths: Description
        """

        X = dataset.X
        if lengths is None:
            lengths = np.asarray([X.shape[0]])

        self.model.fit(X, lengths)
        return
    
    def predict(self, dataset:Dataset, lengths = None) -> OneOrMany[np.ndarray]:
        """
        Docstring for predict
        
        :param self: Description
        :param dataset: Description
        :type dataset: Dataset
        :param lengths: Description
        :return: Description
        :rtype: OneOrMany[ndarray]
        """
        X = dataset.X
        _, state_sequence = self.model.predict(X, lengths)
        return state_sequence # We can not be undoing the transforms over here since Hidden Markov Models are doing Unsupervised Learning and we can not use it for Supervised LearningH
    
    def predict_proba(self, dataset:Dataset, lengths = None) -> np.ndarray:
        """
        Docstring for predict_proba
        
        :param self: Description
        :param dataset: Description
        :type dataset: Dataset
        :param lengths: Description
        :return: Description
        :rtype: ndarray
        """
        X = dataset.X
        _, posteriors = self.model.predict_proba(X, lengths)
        return posteriors
    
    def score_samples(self, dataset:Dataset, lengths = None): # Output is a tuple
        """
        Docstring for score_samples
        
        :param self: Description
        :param dataset: Description
        :type dataset: Dataset
        :param lengths: Description
        """
        X = dataset.X
        return self.model.score_samples(X, lengths)
    
    def score(self, dataset:Dataset, lengths = None) -> float:
        """
        Docstring for score
        
        :param self: Description
        :param dataset: Description
        :type dataset: Dataset
        :param lengths: Description
        :return: Description
        :rtype: float
        """
        X = dataset.X
        return self.model.score(X, lengths)
    
    def decode(self, dataset:Dataset, lengths = None, algorithm = None): # Output is a Tuple
        """
        Docstring for decode
        
        :param self: Description
        :param dataset: Description
        :type dataset: Dataset
        :param lengths: Description
        :param algorithm: Description
        """
        X = dataset.X
        log_prob, state_sequence = self.model.decode(X, lengths, algorithm)
        return log_prob, state_sequence
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