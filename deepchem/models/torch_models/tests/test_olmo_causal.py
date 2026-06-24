import pytest
import deepchem as dc

SMILES = [
    "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1",
]


@pytest.mark.hf
def test_olmo_causal_lm():
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(model="allenai/OLMo-1B-hf",
                 tokenizer=None,
                 task_type="causal_lm")

    dataset = dc.data.NumpyDataset(SMILES)

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None

    generated = model.generate(dataset, max_new_tokens=10)
    assert len(generated) == len(SMILES)
    for text in generated:
        assert isinstance(text, str)
