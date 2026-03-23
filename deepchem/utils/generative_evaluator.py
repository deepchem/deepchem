
"""
Simple utilities to evaluate molecular generative models.

Includes basic metrics:
- validity
- uniqueness
- novelty
- KL divergence (molecular weight-based)
"""

import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import entropy


class MolecularGenerativeEvaluator:
    """
    Evaluator for molecular generative models using SMILES strings.
    """

    def __init__(self, reference_smiles=None, seed=0):
        self.reference_smiles = reference_smiles or []
        self.rng = random.Random(seed)

    @staticmethod
    def mol_from_smiles(s):
        # RDKit returns None for invalid or malformed SMILES
        try:
            return Chem.MolFromSmiles(s)
        except Exception:
            return None

    def canonicalize(self, smiles_list):
        canon = []
        for s in smiles_list:
            mol = self.mol_from_smiles(s)
            if mol is None:
                continue  # skip invalid SMILES
            canon.append(Chem.MolToSmiles(mol, canonical=True))
        return canon

    def validity(self, gen_smiles):
        if not gen_smiles:
            return 0.0

        valid = 0
        for s in gen_smiles:
            if self.mol_from_smiles(s) is not None:
                valid += 1

        return valid / len(gen_smiles)

    def uniqueness(self, gen_smiles):
        if not gen_smiles:
            return 0.0

        canon = self.canonicalize(gen_smiles)
        return len(set(canon)) / len(gen_smiles)

    def novelty(self, gen_smiles):
        if not self.reference_smiles:
            raise ValueError("Reference SMILES required")

        if not gen_smiles:
            return 0.0

        # compare using canonical SMILES to avoid duplicates
        ref_set = set(self.canonicalize(self.reference_smiles))
        gen_canon = self.canonicalize(gen_smiles)

        novel = [s for s in gen_canon if s not in ref_set]
        return len(novel) / len(gen_canon) if gen_canon else 0.0

    def _mw_hist(self, smiles, bins):
        mw = []
        for s in smiles:
            mol = self.mol_from_smiles(s)
            if mol is not None:
                mw.append(Descriptors.MolWt(mol))

        if not mw:
            return None

        hist, _ = np.histogram(mw, bins=bins, density=False)

        # smoothing to avoid zero probabilities in KL divergence
        hist = hist.astype(float) + 1e-8
        hist /= hist.sum()

        return hist

    def kl_divergence(self, gen_smiles, bins=50, sample_size=None):
        if not self.reference_smiles:
            raise ValueError("Reference SMILES required")

        if not gen_smiles:
            return 0.0

        ref = self.reference_smiles

        # sample reference set to avoid bias from ordering
        if sample_size is not None and len(ref) > sample_size:
            ref = self.rng.sample(ref, sample_size)

        gen_hist = self._mw_hist(gen_smiles, bins)
        ref_hist = self._mw_hist(ref, bins)

        if gen_hist is None or ref_hist is None:
            return 0.0

        return float(entropy(gen_hist, ref_hist))

    def evaluate(self, gen_smiles):
        return {
            "validity": self.validity(gen_smiles),
            "uniqueness": self.uniqueness(gen_smiles),
            "novelty": self.novelty(gen_smiles),
            "kl_divergence": self.kl_divergence(gen_smiles),
        }
