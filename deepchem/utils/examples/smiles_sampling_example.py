"""
Example usage of Gibbs-style SMILES sampler.

This demonstrates how to use the run_chain API with a custom step function.

NOTE:
In practice, `gibbs_step_fn` should be implemented using a masked language model.
For example, using HuggingFace Transformers:

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_name = "DeepChem/MoLFormer-c3-1.1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

    def step_fn(smiles):
        return gibbs_step(smiles, model, tokenizer, allowed_tokens)
"""

from deepchem.utils.smiles_sampling import run_chain


def dummy_step(smiles: str) -> str:
    """Identity step (no mutation)."""
    return smiles


seed = "CCO"

trajectory = run_chain(
    seed=seed,
    gibbs_step_fn=dummy_step,
    steps=5
)

print("Generated trajectory:")
for smi in trajectory:
    print(smi)
