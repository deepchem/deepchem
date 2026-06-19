from transformers import AutoModelForCausalLM, AutoTokenizer
import deepchem as dc
def test_huggingface_generate():
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    hf_model = dc.models.torch_models.HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        task="text_generation"
    )

    result = hf_model.generate("DeepChem is", max_length=20)
    assert isinstance(result, list)
    assert isinstance(result[0], str)