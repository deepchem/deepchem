import deepchem as dc
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    "I used a small model as running allenai/OLMo-7B-hf and even running allenai/OLMo-1B-hf takes a long time on CPU. You can change the model to one of those if you have a GPU and want to test out the larger models."
    "However, the smaller distilgpt2 model is still able to generate reasonable text based on the prompt, so it should be sufficient for testing out the HuggingFaceModel's generate function."
    MODEL_NAME = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_model = dc.models.torch_models.HuggingFaceModel(model=model,
                                                       tokenizer=tokenizer,
                                                       task="text_generation")
    prompt = "The molecule has a molecular weight of 300 and a logP of 2.5. It is likely to be"
    answer = hf_model.generate(prompt, max_length=50)
    print(answer[0])


if __name__ == "__main__":
    main()
