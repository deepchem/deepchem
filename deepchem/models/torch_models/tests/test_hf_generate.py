import pytest


@pytest.mark.torch
def test_hugging_face_model_generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepchem.models.torch_models.hf_models import HuggingFaceModel

    torch.manual_seed(123)
    model_id = "sshleifer/tiny-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer)

    prompt = "C1=CC=C"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

    results = hf_model.generate(inputs, max_new_tokens=5)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert results[0].startswith(prompt)
