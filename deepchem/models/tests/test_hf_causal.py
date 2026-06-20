import deepchem as dc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def test_huggingface_causal_lm():
    MODEL_NAME = "distilgpt2"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    hf_model = dc.models.torch_models.HuggingFaceModel(model=model,
                                                       tokenizer=tokenizer,
                                                       task="causal_lm")

    text_list = ["The mitochondria is ", "Water is a "]

    dataset = dc.data.NumpyDataset(text_list)

    # Train
    loss = hf_model.fit(dataset, nb_epoch=1)
    assert loss is not None

    # Predict (reconstruction)
    predictions = hf_model.predict(dataset)
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    assert len(decoded) == len(text_list)
    assert all(isinstance(x, str) for x in decoded)

    print("Decoded reconstruction:", decoded)

    # CPU switch to avoid MPS bug
    inputs = tokenizer(text_list, return_tensors="pt", padding=True)

    # Move model + inputs to CPU for generation
    model_cpu = model.to("cpu")
    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}

    outputs = model_cpu.generate(**inputs_cpu, max_length=20)
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("Generated:", generated)


if __name__ == "__main__":
    test_huggingface_causal_lm()
