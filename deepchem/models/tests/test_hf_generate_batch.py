from transformers import AutoModelForCausalLM, AutoTokenizer
import deepchem as dc
def test_huggingface_generate_batch():
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    hf_model = dc.models.torch_models.HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        task="text_generation"
    )

    prompts = ["Water is a", "DeepChem is a"]
    outputs = hf_model.generate(prompts, batch_size=2, max_length=10)
    assert isinstance(outputs, list)
    assert len(outputs) == 2

if __name__ == "__main__":
    test_huggingface_generate_batch()