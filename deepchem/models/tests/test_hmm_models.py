import numpy as np
from deepchem.wrappers.hmm_models import GaussianHMMWrapper

def test_gaussian_hmm_basic_fit_predict():
    X = np.random.randn(100, 2)

    model = GaussianHMMWrapper(n_components=2, n_iter=5, random_state=42)
    model.fit(X)

    preds = model.predict(X)
    assert len(preds) == len(X)

