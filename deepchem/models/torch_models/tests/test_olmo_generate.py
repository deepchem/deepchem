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
def test_olmo_generate():
    from deepchem.models.torch_models.hf_models import HuggingFaceModel

    MODEL_NAME = "allenai/OLMo-1B-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                              trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16
        if torch.cuda.is_available() else torch.float32,
    )

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                task="causal_lm")

    dataset = dc.data.NumpyDataset(SMILES)
    outputs = hf_model.generate(dataset, max_new_tokens=10)
    assert isinstance(outputs, list)
    assert len(outputs) == len(SMILES)
    for i in outputs:
        assert isinstance(i, str)
        