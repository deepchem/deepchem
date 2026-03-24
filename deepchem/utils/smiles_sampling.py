
import random
from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs


def is_valid_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def tanimoto(s1, s2):
    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)

    if m1 is None or m2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def qed_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)


def gibbs_step(smiles, model, tokenizer, allowed_tokens, k=50):
    """
    Perform one Gibbs-style mutation step using a masked language model.
    """

    tokens = tokenizer.tokenize(smiles)

    safe_positions = [i for i, t in enumerate(tokens) if t in allowed_tokens]

    if not safe_positions:
        return smiles

    pos = random.choice(safe_positions)

    inputs = tokenizer(smiles, return_tensors="pt")
    inputs["input_ids"][0][pos] = tokenizer.mask_token_id

    outputs = model(**inputs)
    logits = outputs.logits[0, pos]

    topk_indices = logits.topk(k).indices

    for idx in topk_indices:

        token = tokenizer.decode([idx])

        if token not in allowed_tokens:
            continue

        new_tokens = tokens.copy()
        new_tokens[pos] = token

        new_smiles = tokenizer.convert_tokens_to_string(new_tokens).replace(" ", "")

        if is_valid_smiles(new_smiles):
            return new_smiles

    return smiles


def run_chain(seed, gibbs_step_fn, steps=50, min_sim=0.3, min_qed=0.4):
    """
    Run constrained Gibbs-style sampling.
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
