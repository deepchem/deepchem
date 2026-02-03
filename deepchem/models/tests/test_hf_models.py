
import importlib.util
import pytest

if importlib.util.find_spec("rdkit") is None:
    pytest.skip("rdkit not installed", allow_module_level=True)

import deepchem as dc
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepchem.models.torch_models.hf_models import HuggingFaceModel


def test_huggingface_generation_from_prompt():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    hf_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        task="generation",
        batch_size=1,
    )

    dataset = dc.data.NumpyDataset(
        X=["The molecule binds to"]
    )

    outputs = hf_model.predict(dataset, max_length=20)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
    assert outputs[0].startswith("The molecule binds to")
