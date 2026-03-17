import deepchem as dc
import torch

from deepchem.models.torch_models.hf_models import OLMoModel

print("DeepChem Version:", dc.__version__)
print("Torch Version:", torch.__version__)

print("\nLoading model...")

model = OLMoModel(
    hf_model_name_or_path="gpt2",
    use_unsloth=False
)

print("Model Loaded Successfully")

prompt = "The molecule CCO is called"

print("\nGenerating text...")

output = model.generate(
    "CC(=O)Oc1ccccc1C(=O)O is", 
    do_sample=True,           # Enable sampling
    top_p=0.9,                # Nucleus sampling
    temperature=0.7,          # Add variety
    repetition_penalty=1.2    # Penalize the loops you saw
)

print("\nGenerated Output:")
print(output)

print("\nTesting SFT training...")

dataset = [
    "CCO is ethanol",
    "CC(=O)O is acetic acid",
    "C is methane"
]

model.sft(
    dataset,
    lora_config=None,
    sft_config={
        "num_train_epochs":1,
        "per_device_train_batch_size":1
    }
)
print("\nSFT Training Completed Successfully")