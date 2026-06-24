import pytest
import deepchem as dc
import numpy as np

SMILES = [
    "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1",
]


@pytest.mark.hf
def test_olmo_single_label_regression():
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(model="allenai/OLMo-1B-hf",
                 tokenizer=None,
                 task_type="regression",
                 n_tasks=1)

    dataset = dc.data.NumpyDataset(SMILES, np.array([[1.0], [0.0]]))

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None

    predictions = model.predict(dataset)
    assert predictions.shape == (len(SMILES), 1)


@pytest.mark.hf
def test_olmo_multi_label_regression():
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(model="allenai/OLMo-1B-hf",
                 tokenizer=None,
                 task_type="regression",
                 n_tasks=2)

    dataset = dc.data.NumpyDataset(SMILES, np.array([[1.0, 0.0], [0.0, 1.0]]))

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None

    predictions = model.predict(dataset)
    assert predictions.shape == (len(SMILES), 2)
