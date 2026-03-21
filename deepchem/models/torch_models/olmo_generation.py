"""
Generation tasks and continued pre-training utilities for OLMo in DeepChem.

This module provides:
- ``OLMoGenerationFeaturizer`` : converts raw molecular strings into
  tokenized tensors suitable for causal-LM training.
- ``OLMoPretrainer`` : a thin wrapper around ``OLMoModel`` that orchestrates
  continued pre-training on molecular datasets (SMILES, SELFIES, IUPAC).
- ``MolecularTextDataset`` : a lightweight ``torch.utils.data.Dataset``
  compatible with DeepChem's ``DiskDataset`` / ``NumpyDataset``.

References
----------
.. [1] Taylor, R., et al. "Galactica: A Large Language Model for Science."
       arXiv:2211.09085 (2022).
.. [2] Fang, Y., et al. "Mol-Instructions: A Large-Scale Biomolecular
       Instruction Dataset for Large Language Models."
       ICLR 2024. https://arxiv.org/abs/2306.08018
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None  # type: ignore

try:
    from transformers import (
        DataCollatorForLanguageModeling,
        PreTrainedTokenizerBase,
        get_cosine_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# DeepChem imports (soft)
try:
    import deepchem as dc
    from deepchem.feat import Featurizer
    from deepchem.models.torch_models.olmo import OLMoModel
except ImportError:
    dc = None  # type: ignore
    Featurizer = object  # type: ignore


# ---------------------------------------------------------------------------
# Featurizer
# ---------------------------------------------------------------------------

class OLMoGenerationFeaturizer(Featurizer):  # type: ignore[misc]
    """Tokenize molecular strings for OLMo causal-LM training and inference.

    Wraps a HuggingFace tokenizer and returns fixed-length integer arrays
    that DeepChem's data pipeline can handle without custom collation.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        An already-loaded HuggingFace tokenizer (e.g. from OLMoModel).
    max_length : int, optional
        Maximum sequence length including EOS.  Default 512.
    add_eos : bool, optional
        Append EOS token after each molecule.  Default ``True``.
    mol_prefix : str, optional
        Optional prefix prepended to each molecule, e.g. ``"SMILES: "``.

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tok = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    >>> feat = OLMoGenerationFeaturizer(tok, max_length=256)
    >>> X = feat.featurize(["CC(=O)O", "c1ccccc1"])
    >>> X.shape
    (2, 256)
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 512,
        add_eos: bool = True,
        mol_prefix: str = "",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos
        self.mol_prefix = mol_prefix

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _featurize(self, mol_string: str) -> np.ndarray:
        text = self.mol_prefix + mol_string
        if self.add_eos:
            text = text + self.tokenizer.eos_token

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return enc["input_ids"][0].astype(np.int64)  # (max_length,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class MolecularTextDataset(Dataset):  # type: ignore[misc]
    """A ``torch.utils.data.Dataset`` over molecular strings for LM training.

    Parameters
    ----------
    molecules : list of str
        SMILES / SELFIES / IUPAC strings.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer to use.
    max_length : int, optional
        Truncation / padding length.
    mol_prefix : str, optional
        String prepended to each molecule before tokenisation.

    Notes
    -----
    Labels are identical to input_ids (standard causal-LM objective).
    Padding positions are set to ``-100`` so they are ignored by
    ``CrossEntropyLoss``.
    """

    molecules: List[str]
    tokenizer: "PreTrainedTokenizerBase" = field(repr=False)
    max_length: int = 512
    mol_prefix: str = ""

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        if idx in self._cache:
            return self._cache[idx]

        text = self.mol_prefix + self.molecules[idx] + self.tokenizer.eos_token
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)         # (L,)
        attention_mask = enc["attention_mask"].squeeze(0)

        # For causal LM: labels == input_ids; mask pad positions
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        self._cache[idx] = item
        return item

    @classmethod
    def from_deepchem_dataset(
        cls,
        dc_dataset: Any,
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 512,
        mol_prefix: str = "",
    ) -> "MolecularTextDataset":
        """Construct from a DeepChem ``NumpyDataset`` or ``DiskDataset``.

        Parameters
        ----------
        dc_dataset : NumpyDataset or DiskDataset
            Must have string features (SMILES / SELFIES) in ``X``.
        tokenizer : PreTrainedTokenizerBase
        max_length : int, optional
        mol_prefix : str, optional

        Returns
        -------
        MolecularTextDataset
        """
        molecules = [str(s) for s in dc_dataset.X.flatten()]
        return cls(
            molecules=molecules,
            tokenizer=tokenizer,
            max_length=max_length,
            mol_prefix=mol_prefix,
        )


# ---------------------------------------------------------------------------
# Pre-trainer
# ---------------------------------------------------------------------------

class OLMoPretrainer:
    """Continued pre-training of OLMo on molecular datasets.

    This class orchestrates a standard causal-LM training loop with:
    - Gradient accumulation
    - Mixed-precision (``torch.cuda.amp``)
    - Cosine LR schedule with linear warm-up
    - Periodic checkpointing

    Parameters
    ----------
    olmo_model : OLMoModel
        An initialised ``OLMoModel`` with ``task="generation"``.
    output_dir : str
        Directory where checkpoints and final weights are saved.
    learning_rate : float, optional
        Peak learning rate (default ``1e-4``).
    batch_size : int, optional
        Per-device training batch size (default 4).
    grad_accum_steps : int, optional
        Gradient accumulation steps (default 8).  Effective batch size =
        ``batch_size * grad_accum_steps``.
    max_epochs : int, optional
        Number of training epochs (default 3).
    warmup_ratio : float, optional
        Fraction of total steps used for LR warm-up (default 0.05).
    mlm_probability : float, optional
        Unused for causal LM; reserved for future masked-LM support.
    fp16 : bool, optional
        Enable AMP float16 (default ``False``).
    bf16 : bool, optional
        Enable AMP bfloat16; preferred on Ampere GPUs (default ``False``).
    save_every_n_steps : int, optional
        Save a checkpoint every N optimiser steps (default 500).
    log_every_n_steps : int, optional
        Log training loss every N steps (default 50).

    Examples
    --------
    >>> from deepchem.models.torch_models.olmo import OLMoModel
    >>> from deepchem.models.torch_models.olmo_generation import OLMoPretrainer, MolecularTextDataset
    >>> model = OLMoModel(task="generation", model_name="allenai/OLMo-1B")
    >>> trainer = OLMoPretrainer(model, output_dir="./olmo_pretrain_ckpt")
    >>> smiles_list = open("pubchem_smiles.smi").read().splitlines()
    >>> trainer.train_on_smiles(smiles_list, max_length=256)
    """

    def __init__(
        self,
        olmo_model: "OLMoModel",
        output_dir: str = "./olmo_pretrain",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        grad_accum_steps: int = 8,
        max_epochs: int = 3,
        warmup_ratio: float = 0.05,
        fp16: bool = False,
        bf16: bool = False,
        save_every_n_steps: int = 500,
        log_every_n_steps: int = 50,
    ) -> None:
        if olmo_model.task != "generation":
            raise ValueError(
                "OLMoPretrainer requires task='generation'. "
                f"Got task='{olmo_model.task}'."
            )

        self.olmo_model = olmo_model
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_epochs = max_epochs
        self.warmup_ratio = warmup_ratio
        self.fp16 = fp16
        self.bf16 = bf16
        self.save_every_n_steps = save_every_n_steps
        self.log_every_n_steps = log_every_n_steps

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_on_smiles(
        self,
        smiles: Sequence[str],
        max_length: int = 512,
        mol_prefix: str = "",
        num_workers: int = 0,
    ) -> List[float]:
        """Run continued pre-training on a list of SMILES strings.

        Parameters
        ----------
        smiles : Sequence[str]
            Molecular SMILES strings to train on.
        max_length : int, optional
            Tokenisation max length (default 512).
        mol_prefix : str, optional
            Optional prefix prepended to each SMILES (default ``""``).
        num_workers : int, optional
            DataLoader workers (default 0 = main process).

        Returns
        -------
        list of float
            Per-step training losses.
        """
        ds = MolecularTextDataset(
            molecules=list(smiles),
            tokenizer=self.olmo_model.tokenizer,
            max_length=max_length,
            mol_prefix=mol_prefix,
        )
        return self._run_training_loop(ds, num_workers=num_workers)

    def train_on_deepchem_dataset(
        self,
        dc_dataset: Any,
        max_length: int = 512,
        mol_prefix: str = "",
        num_workers: int = 0,
    ) -> List[float]:
        """Run continued pre-training on a DeepChem dataset.

        Parameters
        ----------
        dc_dataset : NumpyDataset or DiskDataset
            Dataset whose ``X`` contains SMILES strings.
        max_length : int, optional
        mol_prefix : str, optional
        num_workers : int, optional

        Returns
        -------
        list of float
            Per-step training losses.
        """
        ds = MolecularTextDataset.from_deepchem_dataset(
            dc_dataset,
            tokenizer=self.olmo_model.tokenizer,
            max_length=max_length,
            mol_prefix=mol_prefix,
        )
        return self._run_training_loop(ds, num_workers=num_workers)

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _run_training_loop(
        self,
        dataset: "MolecularTextDataset",
        num_workers: int = 0,
    ) -> List[float]:
        """Core training loop shared by all ``train_on_*`` methods.

        Parameters
        ----------
        dataset : MolecularTextDataset
        num_workers : int

        Returns
        -------
        list of float
        """
        model = self.olmo_model.model
        device = next(model.parameters()).device
        model.train()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        total_steps = len(loader) * self.max_epochs // self.grad_accum_steps
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scaler = None
        amp_dtype = None
        if self.fp16 and device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            amp_dtype = torch.float16
        elif self.bf16 and device.type == "cuda":
            amp_dtype = torch.bfloat16

        all_losses: List[float] = []
        global_step = 0
        optimizer.zero_grad()

        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = 0.0
            for step, batch in enumerate(loader, 1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass (with optional AMP)
                ctx = (
                    torch.cuda.amp.autocast(dtype=amp_dtype)
                    if amp_dtype is not None
                    else _null_context()
                )
                with ctx:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = out.loss / self.grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.grad_accum_steps

                if step % self.grad_accum_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    step_loss = epoch_loss / step
                    all_losses.append(step_loss)

                    if global_step % self.log_every_n_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "Epoch %d | step %d | loss %.4f | lr %.2e",
                            epoch, global_step, step_loss, lr,
                        )

                    if global_step % self.save_every_n_steps == 0:
                        self._save_checkpoint(global_step)

            logger.info(
                "Epoch %d complete. avg loss: %.4f", epoch, epoch_loss / len(loader)
            )

        self._save_checkpoint("final")
        return all_losses

    def _save_checkpoint(self, tag: Union[int, str]) -> None:
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{tag}")
        self.olmo_model.model.save_pretrained(ckpt_dir)
        self.olmo_model.tokenizer.save_pretrained(ckpt_dir)
        logger.info("Checkpoint saved: %s", ckpt_dir)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class _null_context:
    """A no-op context manager (backfill for Python <3.7 contextlib.nullcontext)."""

    def __enter__(self) -> "_null_context":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


def evaluate_generation(
    model: "OLMoModel",
    prompts: Sequence[str],
    *,
    n_per_prompt: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """Compute basic generation quality metrics.

    Parameters
    ----------
    model : OLMoModel
        A fitted ``OLMoModel`` with ``task="generation"``.
    prompts : Sequence[str]
        Conditioning prompts (can be empty strings for unconditional).
    n_per_prompt : int, optional
        How many molecules to generate per prompt.
    max_new_tokens : int, optional
    temperature : float, optional
    top_p : float, optional

    Returns
    -------
    dict
        ``{"generated": [[str]], "unique_ratio": float, "valid_ratio": float}``
        where validity is checked via ``rdkit`` if available.
    """
    generated: List[List[str]] = []
    for prompt in prompts:
        mols = model.generate_molecules(
            prompt=prompt,
            n_molecules=n_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        generated.append(mols)

    flat = [m for batch in generated for m in batch]
    unique_ratio = len(set(flat)) / max(len(flat), 1)

    valid_ratio: Optional[float] = None
    try:
        from rdkit import Chem

        valid = sum(
            1 for s in flat if Chem.MolFromSmiles(s) is not None
        )
        valid_ratio = valid / max(len(flat), 1)
    except ImportError:
        logger.warning("rdkit not installed – skipping validity check.")

    return {
        "generated": generated,
        "unique_ratio": unique_ratio,
        "valid_ratio": valid_ratio,
    }
