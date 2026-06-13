import pytest
import deepchem as dc
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    pass

SMILES = [
    "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1",
]

@pytest.mark.hf
def test_olmo_regression():
    from deepchem.models.torch_models.olmo_class import OlmoClass

    model = OlmoClass()

    dataset = dc.data.NumpyDataset(SMILES, np.array([[1.0], [0.0]]))

    loss = model.regression(dataset, n_tasks=1, nb_epoch=1)
    assert loss is not None

    predictions = model.predict(dataset)
    assert predictions.shape == (len(SMILES), 1)
