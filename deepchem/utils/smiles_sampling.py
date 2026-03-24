"""
Utilities for constrained molecular generation using Gibbs-style sampling.

This module provides:
- SMILES validation
- Similarity computation (Tanimoto)
- QED scoring
- Gibbs-style mutation step using masked language models
- Sampling chains with simple constraints

The implementation is model-agnostic and can be used with any masked language
model for molecular generation (e.g., ChemBERTa-style models).
"""

import random
import torch
from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs


# --- Validation ---
def is_valid_smiles(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


# --- Metrics ---
def tanimoto(s1: str, s2: str) -> float:
    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)

    if m1 is None or m2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def qed_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)


# --- Sampling ---
def gibbs_step(
    smiles: str,
    model,
    tokenizer,
    allowed_tokens: set,
    k: int = 50
) -> str:
    """
    Perform one Gibbs-style mutation step using a masked language model.

    This function is model-agnostic and works with any masked language model
    that supports token masking and logits prediction (e.g., ChemBERTa-style models).

    Args:
        smiles (str): Input SMILES string.
        model: Masked language model.
        tokenizer: Corresponding tokenizer.
        allowed_tokens (set): Tokens allowed for mutation.
        k (int): Top-k sampling size.

    Returns:
        str: Mutated SMILES string (or original if no valid mutation found).
    """

    tokens = tokenizer.tokenize(smiles)

    safe_positions = [i for i, t in enumerate(tokens) if t in allowed_tokens]

    if not safe_positions:
        return smiles

    pos = random.choice(safe_positions)

    inputs = tokenizer(smiles, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    inputs["input_ids"][0][pos] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, pos]
    topk_indices = logits.topk(k).indices

    for idx in topk_indices:
        token = tokenizer.convert_ids_to_tokens(int(idx))

        if token not in allowed_tokens:
            continue

        new_tokens = tokens.copy()
        new_tokens[pos] = token

        new_smiles = tokenizer.convert_tokens_to_string(new_tokens).replace(" ", "")

        if is_valid_smiles(new_smiles):
            return new_smiles

    return smiles


def run_chain(
    seed: str,
    gibbs_step_fn,
    steps: int = 50,
    min_sim: float = 0.3,
    min_qed: float = 0.4
) -> list:
    """
    Run constrained Gibbs-style sampling.

    Applies iterative local mutations using a Gibbs-style proposal function,
    while enforcing similarity and QED constraints.

    Args:
        seed (str): Initial SMILES string.
        gibbs_step_fn (Callable): Function that proposes mutations.
        steps (int): Number of sampling steps.
        min_sim (float): Minimum Tanimoto similarity threshold.
        min_qed (float): Minimum QED threshold.

    Returns:
        list: Trajectory of SMILES strings.
    """

    current = seed
    trajectory = [seed]

    for _ in range(steps):

        proposal = gibbs_step_fn(current)

        if proposal is None or proposal == current:
            trajectory.append(current)
            continue

        sim = tanimoto(seed, proposal)
        q = qed_score(proposal)

        if sim >= min_sim and q >= min_qed:
            current = proposal

        trajectory.append(current)

    return trajectory
