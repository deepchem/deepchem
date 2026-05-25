from transformers import AutoModelForCausalLM, AutoTokenizer
import deepchem as dc
import pandas as pd
def test_huggingface_generate_batch():
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    hf_model = dc.models.torch_models.HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        task="text_generation"
    )
    df = pd.read_csv("datasets/delaney-processed.csv")
    prompts = df["smiles"].tolist()[:2]  # Get the first 2 SMILES strings
    outputs = hf_model.generate(prompts, max_new_tokens=10)
    assert isinstance(outputs, list)
    assert len(outputs) == 2

if __name__ == "__main__":
    test_huggingface_generate_batch()