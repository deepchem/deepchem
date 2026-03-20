import deepchem as dc
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import torch


def continued_pretraining():
    """
Example: Continued pretraining of OLMo on molecular data (Delaney dataset).
Demonstrates dataset integration, checkpointing, and stable generation.
"""
    MODEL_NAME = "allenai/OLMo-1B-hf"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 trust_remote_code=True,
                                                 device_map="auto",
                                                 torch_dtype=torch.float32)

    # Set pad token id in model
    model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing for memory efficiency during training
    model.gradient_checkpointing_enable()

    # Gradient clipping to prevent exploding logits
    for param in model.parameters():
        param.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    # DeepChem wrapper
    hf_model = dc.models.torch_models.HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        task="causal_lm",
        model_dir="./olmo_checkpoints",
        learning_rate=1e-5)

    # Read dataset
    df = pd.read_csv("datasets/delaney-processed.csv")

    # Prepare data
    smiles = df["smiles"].values
    solubility = df["measured log solubility in mols per litre"].values

    # Convert to text format for LLM
    text_list = [
        f"SMILES: {i}. Solubility: {j}." for i, j in zip(smiles, solubility)
    ]

    text_list = text_list[:50]  # subset for quick testing

    # Make labels same as input_ids for causal LM
    dataset = dc.data.NumpyDataset(X=np.array(text_list), y=np.array(text_list))

    # Continued pretraining
    print("Starting training on Delaney dataset...")
    loss = hf_model.fit(dataset, nb_epoch=1, max_checkpoints_to_keep=1)
    print("Training Loss:", loss)

    # Generate after training
    prompts = [
        "SMILES: OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O . Solubility:",
        "SMILES: Cc1occc1C(=O)Nc2ccccc2. Solubility:"
    ]

    # Load from checkpoint safely (avoid GPU OOM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model.model.to("cpu")
    torch.cuda.empty_cache()

    hf_model.load_from_pretrained()

    hf_model.model.to(device)

    # Generate with sampling for more diversity
    with torch.no_grad():
        outputs = hf_model.generate(prompts,
                                    max_length=100,
                                    batch_size=1,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_k=50,
                                    top_p=0.9,
                                    repetition_penalty=1.1,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)

    print("\nGenerated Outputs:")
    for i in outputs:
        print(i)


if __name__ == "__main__":
    continued_pretraining()
