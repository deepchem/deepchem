"""
DeepChem-compatible PyTorch symbolic model.

This module defines DCTorchSymbolicModel, a unified symbolic regression /
classification model implemented in PyTorch and wrapped as a DeepChem
TorchModel. The model learns explicit polynomial symbolic forms over input
features and supports both regression and binary classification tasks.

The symbolic form learned is:

    y = w_lin · x + w_quad · x^2 + b

For classification, the model predicts logits and probabilities are obtained
via sigmoid:

    p = sigmoid(y)

This design allows interpretable symbolic coefficients while remaining fully
compatible with DeepChem training, checkpointing, and evaluation pipelines.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

import numpy as np
import torch
import deepchem as dc

from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss, SigmoidCrossEntropy
from deepchem.models.optimizers import Adam

from .torch_symbolic_net import SymbolicNet

class DCTorchSymbolicModel(TorchModel):
    """
    DeepChem-compatible symbolic model for regression and classification.

    This model learns an explicit symbolic polynomial function over input
    features using PyTorch and exposes it through the DeepChem TorchModel
    interface. It supports both regression and binary classification tasks.

    Parameters
    ----------
    learning_rate : float, default 0.001
        Optimizer learning rate.

    batch_size : int, default 128
        Training batch size.

    task_type : {"regression", "classification"}, default "regression"
        Determines loss function and prediction interpretation.
        - "regression": L2 loss, direct numeric prediction
        - "classification": sigmoid cross-entropy, logits output

    model_dir : str, optional
        Directory for saving/restoring checkpoints.

    Examples
    --------
    Regression example:

    >>> import numpy as np
    >>> import deepchem as dc
    >>> from models.dc_torch_symbolic_model import DCTorchSymbolicModel
    >>>
    >>> X = np.random.randn(100, 3)
    >>> y = 2*X[:,0] - X[:,1] + 0.5
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>>
    >>> model = DCTorchSymbolicModel(task_type="regression")
    >>> model.fit(dataset, nb_epoch=1000)
    >>> preds = model.predict(dataset)

    Classification example:

    >>> X = np.random.randn(200, 2)
    >>> y = (X[:,0] + X[:,1] > 0).astype(float)
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>>
    >>> model = DCTorchSymbolicModel(task_type="classification")
    >>> model.fit(dataset, nb_epoch=1000)
    >>> logits = model.predict(dataset)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        task_type: str = "regression",
        model_dir: Optional[str] = None,
    ) -> None:

        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")

        self.task_type = task_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # dummy model required by DeepChem init
        dummy = torch.nn.Linear(1, 1)

        loss = L2Loss() if task_type == "regression" else SigmoidCrossEntropy()

        super().__init__(
            model=dummy,
            loss=loss,
            optimizer=Adam(learning_rate=learning_rate, weight_decay=1e-4),
            batch_size=batch_size,
            output_types=["prediction"],
            model_dir=model_dir,
        )

        self._built_symbolic = False

    # --------------------------------------------------
    # Build symbolic architecture once feature count known
    # --------------------------------------------------

    def build(self, dataset: dc.data.Dataset) -> None:
        """
        Construct symbolic network using dataset feature dimension.
        """
        n_features = dataset.X.shape[1]
        self.model = SymbolicNet(n_features)

        self._built_symbolic = True
        self._built = False
        self._ensure_built()

    # --------------------------------------------------
    # Fit wrapper
    # --------------------------------------------------

    def fit(self, dataset: dc.data.Dataset, **kwargs: Any) -> float:
        """
        Train symbolic model on dataset.
        """
        if not self._built_symbolic:
            self.build(dataset)
        return super().fit(dataset, **kwargs)

    # --------------------------------------------------
    # Equation extraction
    # --------------------------------------------------

    def get_equation(self) -> Dict[str, np.ndarray]:
        """
        Return learned symbolic coefficients.

        Returns
        -------
        dict
            Dictionary with keys:
            - "linear"
            - "quadratic"
            - "bias"
        """
        eq = self.model.get_equation()

        if isinstance(eq, tuple):
            w, b = eq
            return {
                "linear": w,
                "quadratic": np.zeros_like(w),
                "bias": b,
            }

        eq.setdefault("linear", None)
        eq.setdefault("quadratic", None)
        eq.setdefault("bias", 0.0)

        return eq

    # --------------------------------------------------
    # Save wrapper
    # --------------------------------------------------

    def save_checkpoint(
        self,
        max_checkpoints_to_keep: int | str = 5,
        model_dir: Optional[str] = None,
    ) -> None:
        """
        Save model checkpoint.

        Supports both DeepChem internal calls and direct usage:

        >>> model.save_checkpoint("path")
        """
        if isinstance(max_checkpoints_to_keep, str) and model_dir is None:
            model_dir = max_checkpoints_to_keep
            max_checkpoints_to_keep = 5

        super().save_checkpoint(
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            model_dir=model_dir,
        )

    # --------------------------------------------------
    # Restore symbolic model correctly
    # --------------------------------------------------

    def restore(self, checkpoint: Optional[str] = None) -> None:
        """
        Restore symbolic model from checkpoint.
        """
        if not self._built_symbolic:
            if checkpoint is None:
                checkpoint = self.get_checkpoints()[-1]

            data = torch.load(checkpoint, map_location=self.device)
            state = data["model_state_dict"]

            # infer feature count
            w = state["linear.weight"]
            n_features = w.shape[1]

            self.model = SymbolicNet(n_features)
            self._built_symbolic = True
            self._built = False
            self._ensure_built()

        super().restore(checkpoint)