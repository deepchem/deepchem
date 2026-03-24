"""
Example usage of Gibbs-style SMILES sampler.

This demonstrates how to run constrained sampling and inspect
trajectory behavior such as stability and diversity.

NOTE:
In practice, `gibbs_step_fn` should be implemented using a masked
language model. See the commented section below for integration
with models like MoLFormer.
"""

from deepchem.utils.smiles_sampling import run_chain


def dummy_step(smiles: str) -> str:
    """
    Identity step (no mutation).

    Replace this with a model-based step function.
    """
    return smiles


seed = "CCO"

trajectory = run_chain(
    seed=seed,
    gibbs_step_fn=dummy_step,
    steps=10
)

print("Generated trajectory:")
for smi in trajectory:
    print(smi)

print("\nUnique molecules:", len(set(trajectory)))


# ---- Advanced usage (commented) ----
# Example with a masked language model:
#
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from deepchem.utils.smiles_sampling import gibbs_step
#
# model_name = "DeepChem/MoLFormer-c3-1.1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
#
# allowed_tokens = set(tokenizer.get_vocab().keys())
#
# def step_fn(smiles):
#     return gibbs_step(smiles, model, tokenizer, allowed_tokens)
#
# trajectory = run_chain("CCO", step_fn, steps=20)
