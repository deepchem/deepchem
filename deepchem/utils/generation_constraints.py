"""
Chemically-constrained logits processors for autoregressive
SMILES generation with large language models.

These processors integrate into HuggingFace's LogitsProcessor
API to enforce chemical validity at every decoding step,
preventing the BPE tokenizer blindness problem observed when
applying causal LLMs to molecular generation tasks.

Classes
-------
RDKitGuidedTSM
    Boosts EOS token probability when a valid molecule is
    formed at each decoding step using RDKit validation.
HybridTSM
    Combines Markov grammar rules as a fast pre-filter with
    RDKit validation for accurate topological checking.

References
----------
Ramsundar et al., Deep Learning for the Life Sciences, 2019.
"""

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from rdkit import Chem, RDLogger
from typing import List, Optional
import warnings

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

__all__ = ["RDKitGuidedTSM", "HybridTSM"]


class RDKitGuidedTSM(LogitsProcessor):
    """Boost EOS when RDKit validates current sequence as complete molecule.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used with the language model.
    eos_boost : float, optional (default=50.0)
        Score added to EOS token when valid molecule is detected.

    Examples
    --------
    >>> from transformers import LogitsProcessorList
    >>> processor = RDKitGuidedTSM(tokenizer, eos_boost=50.0)
    >>> processor_list = LogitsProcessorList([processor])
    """

    def __init__(self, tokenizer, eos_boost: float = 50.0):
        self.tokenizer = tokenizer
        self.eos_boost = eos_boost
        self.eos_id = tokenizer.eos_token_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply RDKit-guided EOS boosting at each decoding step.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Token ids generated so far, shape (batch, seq_len).
        scores : torch.FloatTensor
            Next token logits, shape (batch, vocab_size).

        Returns
        -------
        torch.FloatTensor
            Modified scores with EOS boosted when valid molecule found.
        """
        for i in range(input_ids.shape[0]):
            seq = self.tokenizer.decode(
                input_ids[i],
                skip_special_tokens=True
            )
            if seq and len(seq.strip()) > 0:
                mol = Chem.MolFromSmiles(seq, sanitize=True)
                if mol is not None and mol.GetNumAtoms() > 0:
                    scores[i, self.eos_id] += self.eos_boost
        return scores


class HybridTSM(LogitsProcessor):
    """Markov grammar rules as fast pre-filter combined with RDKit validation.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used with the language model.
    penalty : float, optional (default=-1e9)
        Score assigned to invalid tokens to block them.
    eos_boost : float, optional (default=50.0)
        Score added to EOS token when valid molecule is detected.

    Examples
    --------
    >>> from transformers import LogitsProcessorList
    >>> processor = HybridTSM(tokenizer, penalty=-1e9, eos_boost=50.0)
    >>> processor_list = LogitsProcessorList([processor])
    """

    def __init__(
        self,
        tokenizer,
        penalty: float = -1e9,
        eos_boost: float = 50.0
    ):
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.eos_boost = eos_boost
        self.eos_id = tokenizer.eos_token_id

        self.atoms = {
            "C", "c", "N", "n", "O", "o", "S", "s",
            "F", "P", "p", "B", "b", "Cl", "Br", "I", "H"
        }
        self.bonds = {"-", "=", "#"}
        self.digits = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        self.openers = {"(", "["}
        self.closers = {")", "]"}

        all_valid = (
            self.atoms | self.bonds |
            self.digits | self.openers | self.closers
        )

        self.allowed_ids = set()
        for tid in range(tokenizer.vocab_size):
            decoded = tokenizer.decode([tid])
            if decoded in all_valid:
                self.allowed_ids.add(tid)
        self.allowed_ids.add(self.eos_id)

    def _markov_rules(
        self,
        scores: torch.FloatTensor,
        seq: str,
        i: int
    ) -> torch.FloatTensor:
        """Apply Markov grammar rules to block structurally invalid tokens.

        Parameters
        ----------
        scores : torch.FloatTensor
            Current logits for sequence i.
        seq : str
            Decoded SMILES string generated so far.
        i : int
            Batch index.

        Returns
        -------
        torch.FloatTensor
            Scores with invalid transitions penalized.
        """
        if not seq:
            return scores

        last = seq[-1]
        open_branches = seq.count("(") - seq.count(")")
        open_rings = [
            d for d in "123456789"
            if seq.count(d) % 2 != 0
        ]

        last_is_bond = last in self.bonds
        last_is_opener = last in self.openers
        last_is_digit = last in self.digits

        for tid in self.allowed_ids:
            tok = self.tokenizer.decode([tid])
            if tid == self.eos_id:
                tok = "<EOS>"

            if last_is_bond:
                if tok in self.bonds or tok in self.digits:
                    scores[i, tid] = self.penalty
                if tok in self.closers or tok == "<EOS>":
                    scores[i, tid] = self.penalty

            if last_is_opener:
                if tok in self.digits or tok in self.openers:
                    scores[i, tid] = self.penalty
                if tok in self.closers or tok == "<EOS>":
                    scores[i, tid] = self.penalty

            if last_is_digit:
                if tok in self.digits:
                    scores[i, tid] = self.penalty

            if tok == ")" and open_branches <= 0:
                scores[i, tid] = self.penalty

            if tok == "<EOS>":
                if (open_branches > 0
                        or len(open_rings) > 0
                        or last_is_bond):
                    scores[i, tid] = self.penalty

        return scores

    def _rdkit_boost(
        self,
        scores: torch.FloatTensor,
        seq: str,
        i: int
    ) -> torch.FloatTensor:
        """Boost EOS if current sequence is a valid molecule via RDKit.

        Parameters
        ----------
        scores : torch.FloatTensor
            Current logits for sequence i.
        seq : str
            Decoded SMILES string generated so far.
        i : int
            Batch index.

        Returns
        -------
        torch.FloatTensor
            Scores with EOS boosted if valid molecule detected.
        """
        if seq and len(seq.strip()) > 0:
            mol = Chem.MolFromSmiles(seq, sanitize=True)
            if mol is not None and mol.GetNumAtoms() > 0:
                scores[i, self.eos_id] += self.eos_boost
        return scores

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply HybridTSM constraints at each decoding step.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Token ids generated so far, shape (batch, seq_len).
        scores : torch.FloatTensor
            Next token logits, shape (batch, vocab_size).

        Returns
        -------
        torch.FloatTensor
            Scores after Markov pre-filter and RDKit boost applied.
        """
        for i in range(scores.shape[0]):
            mask = torch.ones(scores.shape[1], dtype=torch.bool)
            mask[list(self.allowed_ids)] = False
            scores[i, mask] = self.penalty

        for i in range(input_ids.shape[0]):
            seq = self.tokenizer.decode(
                input_ids[i],
                skip_special_tokens=True
            )
            scores = self._markov_rules(scores, seq, i)

        for i in range(input_ids.shape[0]):
            seq = self.tokenizer.decode(
                input_ids[i],
                skip_special_tokens=True
            )
            scores = self._rdkit_boost(scores, seq, i)

        return scores
