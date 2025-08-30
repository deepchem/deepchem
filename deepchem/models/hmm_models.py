import deepchem as dc
from deepchem.models import Model
from hmmlearn.hmm import GaussianHMM

class HMMModel(Model):
    """DeepChem wrapper for hmmlearn models."""

    def __init__(self, n_components=2, model_type="gaussian", **kwargs):
        super(HMMModel, self).__init__(model_dir=None)

        if model_type == "gaussian":
            self.model = GaussianHMM(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, dataset, **kwargs):
        """Fit HMM on dataset (expects sequences as X)."""
        X = dataset.X
        self.model.fit(X, **kwargs)

    def predict(self, dataset):
        """Predict hidden states for sequences."""
        X = dataset.X
        return self.model.predict(X)

    def score(self, dataset):
        """Return log-likelihood of the data under the model."""
        X = dataset.X
        return self.model.score(X)
