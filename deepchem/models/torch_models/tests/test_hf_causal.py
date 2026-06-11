import pytest
import deepchem as dc

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    pass

SMILES = [
    "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1",
]


@pytest.mark.hf
def test_hf_causal():
    from deepchem.models.torch_models.hf_models import HuggingFaceModel

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                task="causal_lm")

    dataset = dc.data.NumpyDataset(SMILES)

    loss = hf_model.fit(dataset, nb_epoch=1)
    assert loss is not None

    predictions = hf_model.predict(dataset)
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    assert len(decoded) == len(SMILES)
    for i in decoded:
        assert isinstance(i, str)

    generated = hf_model.generate(dataset, max_new_tokens=10)
    assert len(generated) == len(SMILES)
    for j in generated:
        assert isinstance(j, str)