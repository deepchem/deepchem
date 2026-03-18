import torch
import deepchem as dc
from transformers import AutoModelForCausalLM, AutoTokenizer


def olmo_generate():
    """Example of using HuggingFaceModel to generate text using an OLMo model with DeepChem's HuggingFaceModel class."""

    MODEL_NAME = "allenai/OLMo-1B-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                              trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
        if torch.cuda.is_available() else torch.float32)

    hf_model = dc.models.torch_models.HuggingFaceModel(model=model,
                                                       tokenizer=tokenizer,
                                                       task="text_generation")

    prompts = [
        "The molecule has a molecular weight of 300 and a logP of 2.5. It is likely to be",
        "Water is a"
    ]
    answers = hf_model.generate(prompts, batch_size=2, max_length=50)

    for i in answers:
        print(i)


if __name__ == "__main__":
    olmo_generate()
