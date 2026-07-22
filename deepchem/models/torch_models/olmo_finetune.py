"""
Classification and Regression fine-tuning utilities for OLMo in DeepChem.

This module provides:
- ``OLMoFineTuner`` : unified fine-tuning orchestrator for classification and
  regression tasks built on top of ``OLMoModel``.
- ``OLMoMolecularFeaturizer`` : converts SMILES + labels into tokenized
  (input_ids, attention_mask, label) tuples for supervised training.
- Helper functions for evaluation: ``evaluate_classification``,
  ``evaluate_regression``.

References
----------
.. [1] Hu, E.J., et al. "LoRA: Low-Rank Adaptation of Large Language Models."
       ICLR 2022. https://arxiv.org/abs/2106.09685
.. [2] Chithrananda, S., et al. "ChemBERTa: Large-Scale Self-Supervised
       Pretraining for Molecular Property Prediction."
       arXiv:2010.09885 (2020).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from deepchem.models.torch_models.olmo import OLMoModel
except ImportError:
    OLMoModel = object  # type: ignore


# ---------------------------------------------------------------------------
# Supervised dataset
# ---------------------------------------------------------------------------

class OLMoSupervisedDataset(Dataset):  # type: ignore[misc]
    """Tokenised dataset for classification / regression fine-tuning.

    Parameters
    ----------
    smiles : Sequence[str]
        Molecular SMILES strings.
    labels : np.ndarray
        Ground-truth labels.  Shape ``(n_samples, n_tasks)``.
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    max_length : int, optional
        Maximum tokenisation length (default 256).
    task : str, optional
        ``"classification"`` or ``"regression"``.
    smiles_prefix : str, optional
        Optional prefix before each SMILES, e.g. ``"Predict property: "``.

    Examples
    --------
    >>> ds = OLMoSupervisedDataset(smiles, labels, tokenizer, task="classification")
    >>> loader = DataLoader(ds, batch_size=8)
    """

    def __init__(
        self,
        smiles: Sequence[str],
        labels: np.ndarray,
        tokenizer: Any,
        max_length: int = 256,
        task: str = "classification",
        smiles_prefix: str = "",
    ) -> None:
        self.smiles = list(smiles)
        self.labels = np.asarray(labels)
        if self.labels.ndim == 1:
            self.labels = self.labels[:, None]  # (N, 1)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.smiles_prefix = smiles_prefix

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        text = self.smiles_prefix + self.smiles[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label,
        }

    @classmethod
    def from_deepchem(
        cls,
        dc_dataset: Any,
        tokenizer: Any,
        max_length: int = 256,
        task: str = "classification",
        smiles_prefix: str = "",
    ) -> "OLMoSupervisedDataset":
        """Build from a DeepChem dataset.

        Parameters
        ----------
        dc_dataset : NumpyDataset or DiskDataset
            ``X`` must contain SMILES strings; ``y`` contains labels.
        tokenizer : PreTrainedTokenizerBase
        max_length : int, optional
        task : str, optional
        smiles_prefix : str, optional
        """
        smiles = [str(s) for s in dc_dataset.X.flatten()]
        labels = dc_dataset.y
        return cls(smiles, labels, tokenizer, max_length, task, smiles_prefix)


# ---------------------------------------------------------------------------
# Fine-tuner
# ---------------------------------------------------------------------------

class OLMoFineTuner:
    """High-level fine-tuning loop for OLMo classification / regression.

    Supports:
    - Single-task and multi-task settings.
    - Binary cross-entropy (classification) and MSE / Huber (regression).
    - Weighted sampling for imbalanced datasets.
    - Periodic evaluation on a validation split.

    Parameters
    ----------
    olmo_model : OLMoModel
        An ``OLMoModel`` configured with ``task="classification"`` or
        ``task="regression"``.
    output_dir : str
        Directory for checkpoints and best-model weights.
    learning_rate : float, optional
        Peak AdamW learning rate (default ``2e-5``).
    batch_size : int, optional
        Per-device batch size (default 8).
    max_epochs : int, optional
        Training epochs (default 5).
    grad_accum_steps : int, optional
        Gradient accumulation (default 1).
    weight_decay : float, optional
        AdamW weight decay (default ``0.01``).
    warmup_ratio : float, optional
        LR warm-up fraction (default ``0.06``).
    loss_fn : str, optional
        Loss function name.  Auto-selected from task if not specified.
        Options: ``"bce"``, ``"ce"``, ``"mse"``, ``"huber"``.
    class_weights : Sequence[float] or None, optional
        Per-class weights for ``"bce"`` / ``"ce"`` losses.
    fp16 : bool, optional
        Enable AMP (default ``False``).
    bf16 : bool, optional
        Enable BF16 AMP (default ``False``).
    save_best : bool, optional
        If ``True``, save the checkpoint with the best validation metric.
    early_stopping_patience : int, optional
        Stop training if validation metric does not improve for this many
        epochs (default ``-1`` means disabled).

    Examples
    --------
    Classification:

    >>> model = OLMoModel(task="classification", n_tasks=1, n_classes=2,
    ...                   use_lora=True)
    >>> tuner = OLMoFineTuner(model, output_dir="./ckpt_cls", max_epochs=5)
    >>> tuner.fit(train_smiles, train_labels,
    ...           val_smiles=val_smiles, val_labels=val_labels)

    Regression:

    >>> model = OLMoModel(task="regression", n_tasks=1, use_lora=True)
    >>> tuner = OLMoFineTuner(model, output_dir="./ckpt_reg",
    ...                       loss_fn="huber", max_epochs=5)
    >>> tuner.fit(train_smiles, train_labels)
    """

    def __init__(
        self,
        olmo_model: "OLMoModel",
        output_dir: str = "./olmo_finetune",
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        max_epochs: int = 5,
        grad_accum_steps: int = 1,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        loss_fn: Optional[str] = None,
        class_weights: Optional[Sequence[float]] = None,
        fp16: bool = False,
        bf16: bool = False,
        save_best: bool = True,
        early_stopping_patience: int = -1,
    ) -> None:
        if olmo_model.task not in ("classification", "regression"):
            raise ValueError(
                "OLMoFineTuner requires task in ('classification', 'regression')."
            )

        self.olmo_model = olmo_model
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.grad_accum_steps = grad_accum_steps
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.fp16 = fp16
        self.bf16 = bf16
        self.save_best = save_best
        self.early_stopping_patience = early_stopping_patience

        os.makedirs(output_dir, exist_ok=True)

        # Resolve loss function
        self._loss_fn = loss_fn or (
            "bce" if olmo_model.task == "classification" else "mse"
        )
        self._class_weights = class_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        smiles: Sequence[str],
        labels: Union[np.ndarray, List],
        val_smiles: Optional[Sequence[str]] = None,
        val_labels: Optional[Union[np.ndarray, List]] = None,
        max_length: int = 256,
        smiles_prefix: str = "",
        num_workers: int = 0,
    ) -> Dict[str, List[float]]:
        """Fine-tune the model on labelled SMILES data.

        Parameters
        ----------
        smiles : Sequence[str]
            Training SMILES.
        labels : array-like
            Training labels.  Shape ``(N,)`` or ``(N, n_tasks)``.
        val_smiles : Sequence[str], optional
            Validation SMILES for early stopping / best model.
        val_labels : array-like, optional
            Validation labels.
        max_length : int, optional
            Tokenisation length (default 256).
        smiles_prefix : str, optional
        num_workers : int, optional

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_metric": [...]}``
        """
        labels = np.asarray(labels)
        train_ds = OLMoSupervisedDataset(
            smiles, labels, self.olmo_model.tokenizer,
            max_length=max_length,
            task=self.olmo_model.task,
            smiles_prefix=smiles_prefix,
        )
        val_ds = None
        if val_smiles is not None and val_labels is not None:
            val_ds = OLMoSupervisedDataset(
                val_smiles,
                np.asarray(val_labels),
                self.olmo_model.tokenizer,
                max_length=max_length,
                task=self.olmo_model.task,
                smiles_prefix=smiles_prefix,
            )

        return self._run_training(train_ds, val_ds, num_workers)

    def fit_deepchem(
        self,
        train_dc: Any,
        val_dc: Optional[Any] = None,
        max_length: int = 256,
        smiles_prefix: str = "",
        num_workers: int = 0,
    ) -> Dict[str, List[float]]:
        """Fine-tune from DeepChem ``NumpyDataset`` objects.

        Parameters
        ----------
        train_dc : NumpyDataset
        val_dc : NumpyDataset, optional
        max_length : int, optional
        smiles_prefix : str, optional
        num_workers : int, optional

        Returns
        -------
        dict
        """
        train_ds = OLMoSupervisedDataset.from_deepchem(
            train_dc, self.olmo_model.tokenizer,
            max_length=max_length, task=self.olmo_model.task,
            smiles_prefix=smiles_prefix,
        )
        val_ds = None
        if val_dc is not None:
            val_ds = OLMoSupervisedDataset.from_deepchem(
                val_dc, self.olmo_model.tokenizer,
                max_length=max_length, task=self.olmo_model.task,
                smiles_prefix=smiles_prefix,
            )
        return self._run_training(train_ds, val_ds, num_workers)

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _build_criterion(self, device: "torch.device") -> nn.Module:
        weights = None
        if self._class_weights is not None:
            weights = torch.tensor(self._class_weights, dtype=torch.float32).to(device)

        if self._loss_fn == "bce":
            return nn.BCEWithLogitsLoss(pos_weight=weights)
        elif self._loss_fn == "ce":
            return nn.CrossEntropyLoss(weight=weights)
        elif self._loss_fn == "mse":
            return nn.MSELoss()
        elif self._loss_fn == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss_fn: {self._loss_fn!r}")

    def _forward_batch(
        self,
        model: nn.Module,
        batch: Dict[str, "torch.Tensor"],
        device: "torch.device",
    ) -> "torch.Tensor":
        """Run a forward pass and return raw logits / predictions."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if self.olmo_model.task == "classification":
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits  # (B, n_classes * n_tasks)
        else:
            # Regression: use pooled hidden state + regression head
            return self.olmo_model._pooled_hidden_state(input_ids, attention_mask)

    def _run_training(
        self,
        train_ds: OLMoSupervisedDataset,
        val_ds: Optional[OLMoSupervisedDataset],
        num_workers: int,
    ) -> Dict[str, List[float]]:
        from transformers import get_cosine_schedule_with_warmup

        model = self.olmo_model.model
        device = next(model.parameters()).device
        model.train()
        criterion = self._build_criterion(device)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * self.max_epochs // self.grad_accum_steps
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        amp_dtype = None
        scaler = None
        if self.fp16 and device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            amp_dtype = torch.float16
        elif self.bf16 and device.type == "cuda":
            amp_dtype = torch.bfloat16

        history: Dict[str, List[float]] = {"train_loss": [], "val_metric": []}
        best_val = float("inf") if self.olmo_model.task == "regression" else -float("inf")
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, 1):
                labels = batch["labels"].to(device)

                ctx = (
                    torch.cuda.amp.autocast(dtype=amp_dtype)
                    if amp_dtype else _null_ctx()
                )
                with ctx:
                    preds = self._forward_batch(model, batch, device)
                    if self.olmo_model.task == "classification":
                        loss = criterion(preds, labels) / self.grad_accum_steps
                    else:
                        loss = criterion(preds, labels) / self.grad_accum_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item() * self.grad_accum_steps

                if step % self.grad_accum_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = running_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            logger.info("Epoch %d | train_loss: %.4f", epoch, avg_loss)

            if val_ds is not None:
                val_metric = self._evaluate(val_ds, device, num_workers=0)
                history["val_metric"].append(val_metric)
                logger.info("Epoch %d | val_metric: %.4f", epoch, val_metric)

                improved = (
                    val_metric < best_val
                    if self.olmo_model.task == "regression"
                    else val_metric > best_val
                )
                if improved:
                    best_val = val_metric
                    patience_counter = 0
                    if self.save_best:
                        self._save("best")
                else:
                    patience_counter += 1
                    if (
                        self.early_stopping_patience > 0
                        and patience_counter >= self.early_stopping_patience
                    ):
                        logger.info("Early stopping triggered at epoch %d.", epoch)
                        break

        self._save("final")
        return history

    def _evaluate(
        self,
        ds: OLMoSupervisedDataset,
        device: "torch.device",
        num_workers: int = 0,
    ) -> float:
        """Return a scalar validation metric (AUROC or RMSE)."""
        loader = DataLoader(ds, batch_size=self.batch_size, num_workers=num_workers)
        model = self.olmo_model.model
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                preds = self._forward_batch(model, batch, device)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch["labels"].numpy())

        preds_arr = np.concatenate(all_preds, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)

        if self.olmo_model.task == "classification":
            probs = torch.sigmoid(torch.tensor(preds_arr)).numpy()
            if HAS_SKLEARN:
                try:
                    return float(roc_auc_score(labels_arr.flatten(), probs.flatten()))
                except Exception:
                    pass
            return float(np.mean((probs > 0.5).astype(float) == labels_arr))

        # Regression → RMSE
        rmse = float(np.sqrt(mean_squared_error(labels_arr, preds_arr))) if HAS_SKLEARN else float(
            np.sqrt(np.mean((preds_arr - labels_arr) ** 2))
        )
        return rmse

    def _save(self, tag: Union[str, int]) -> None:
        ckpt = os.path.join(self.output_dir, f"checkpoint-{tag}")
        self.olmo_model.model.save_pretrained(ckpt)
        self.olmo_model.tokenizer.save_pretrained(ckpt)
        logger.info("Checkpoint saved: %s", ckpt)


# ---------------------------------------------------------------------------
# Standalone evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_classification(
    model: "OLMoModel",
    smiles: Sequence[str],
    labels: Union[np.ndarray, List],
    max_length: int = 256,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a fine-tuned classification model.

    Parameters
    ----------
    model : OLMoModel
        Trained classification model.
    smiles : Sequence[str]
    labels : array-like
        Ground-truth binary labels.
    max_length : int, optional
    batch_size : int, optional
    threshold : float, optional
        Decision threshold for accuracy (default 0.5).

    Returns
    -------
    dict
        ``{"roc_auc": float, "prc_auc": float, "accuracy": float}``
    """
    ds = OLMoSupervisedDataset(
        smiles, np.asarray(labels), model.tokenizer,
        max_length=max_length, task="classification",
    )
    loader = DataLoader(ds, batch_size=batch_size)
    device = next(model.model.parameters()).device
    model.model.eval()

    preds_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            iids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model.model(input_ids=iids, attention_mask=mask).logits
            preds_list.append(torch.sigmoid(logits).cpu().numpy())
            labels_list.append(batch["labels"].numpy())

    preds = np.concatenate(preds_list).flatten()
    true = np.concatenate(labels_list).flatten()

    results: Dict[str, float] = {}
    results["accuracy"] = float(np.mean((preds > threshold) == true))

    if HAS_SKLEARN:
        try:
            results["roc_auc"] = float(roc_auc_score(true, preds))
            results["prc_auc"] = float(average_precision_score(true, preds))
        except Exception as e:
            logger.warning("sklearn metric failed: %s", e)

    return results


def evaluate_regression(
    model: "OLMoModel",
    smiles: Sequence[str],
    labels: Union[np.ndarray, List],
    max_length: int = 256,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Evaluate a fine-tuned regression model.

    Parameters
    ----------
    model : OLMoModel
    smiles : Sequence[str]
    labels : array-like
    max_length : int, optional
    batch_size : int, optional

    Returns
    -------
    dict
        ``{"rmse": float, "mae": float, "r2": float}``
    """
    ds = OLMoSupervisedDataset(
        smiles, np.asarray(labels), model.tokenizer,
        max_length=max_length, task="regression",
    )
    loader = DataLoader(ds, batch_size=batch_size)
    device = next(model.model.parameters()).device
    model.model.eval()

    preds_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            iids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            preds = model._pooled_hidden_state(iids, mask)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(batch["labels"].numpy())

    preds_arr = np.concatenate(preds_list)
    true_arr = np.concatenate(labels_list)

    results: Dict[str, float] = {}
    if HAS_SKLEARN:
        results["rmse"] = float(np.sqrt(mean_squared_error(true_arr, preds_arr)))
        results["mae"] = float(mean_absolute_error(true_arr, preds_arr))
        results["r2"] = float(r2_score(true_arr, preds_arr))
    else:
        diff = preds_arr - true_arr
        results["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        results["mae"] = float(np.mean(np.abs(diff)))

    return results


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass
