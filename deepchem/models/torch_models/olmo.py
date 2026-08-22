"""
OLMo (Open Language Model) integration for DeepChem via HuggingFaceModel wrapper.

This module provides the core OLMoModel class that wraps allenai/OLMo-7B (and
other OLMo variants) using DeepChem's HuggingFaceModel abstraction.

References
----------
.. [1] Groeneveld, D., et al. "OLMo: Accelerating the Science of Language Models."
       arXiv:2402.00838 (2024). https://arxiv.org/abs/2402.00838
.. [2] HuggingFace OLMo: https://huggingface.co/allenai/OLMo-7B

Example
-------
>>> from deepchem.models.torch_models.olmo import OLMoModel
>>> model = OLMoModel(task="generation", model_name="allenai/OLMo-7B")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "transformers not found. Install via: pip install transformers>=4.40.0"
    )

try:
    from deepchem.models.torch_models.hf_models import HuggingFaceModel
    HAS_DEEPCHEM_HF = True
except ImportError:
    HAS_DEEPCHEM_HF = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLMO_SUPPORTED_TASKS = ("generation", "classification", "regression")

OLMO_MODEL_REGISTRY: Dict[str, str] = {
    "olmo-1b": "allenai/OLMo-1B",
    "olmo-7b": "allenai/OLMo-7B",
    "olmo-7b-instruct": "allenai/OLMo-7B-Instruct",
}

_DEFAULT_MODEL = "allenai/OLMo-7B"
_DEFAULT_MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Head modules
# ---------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """A simple MLP regression head appended on top of OLMo's hidden states.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the last hidden layer of OLMo.
    n_tasks : int, optional (default 1)
        Number of regression targets.
    dropout : float, optional (default 0.1)
        Dropout probability applied before the linear projection.
    """

    def __init__(
        self,
        hidden_size: int,
        n_tasks: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, n_tasks)
        self.activation = nn.GELU()

    def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(batch, hidden_size)`` – typically the last-token or
            mean-pooled representation.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_tasks)``.
        """
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class OLMoModel(HuggingFaceModel if HAS_DEEPCHEM_HF else object):  # type: ignore[misc]
    """DeepChem wrapper around the OLMo family of causal language models.

    OLMo (Open Language Model) by Allen AI is a fully open-source LLM whose
    weights, training data, and code are publicly available.  This wrapper
    exposes OLMo for three downstream use-cases common in cheminformatics:

    * **generation** – unconditional / conditional molecular SMILES / SELFIES
      generation (causal LM head unchanged).
    * **classification** – binary or multi-class property prediction via a
      linear head on top of the final token representation.
    * **regression** – continuous property prediction (e.g. logP, IC50) via a
      small MLP head.

    It also supports **continued pre-training** on molecular corpora through
    the standard ``fit`` interface with ``task="generation"``.

    Parameters
    ----------
    task : str
        One of ``"generation"``, ``"classification"``, or ``"regression"``.
    model_name : str, optional
        HuggingFace Hub model identifier.  Defaults to ``"allenai/OLMo-7B"``.
        Any OLMo checkpoint (1B, 7B, instruct) is supported.
    n_tasks : int, optional
        Number of output targets.  Only used for classification / regression.
    n_classes : int, optional
        Number of classes per task.  Only used when ``task="classification"``.
    tokenizer_path : str or None, optional
        Path / Hub ID for a custom tokenizer.  When ``None`` the tokenizer is
        loaded from ``model_name``.
    max_seq_length : int, optional
        Maximum tokenisation length (default 512).
    use_lora : bool, optional
        If ``True`` and ``peft`` is installed, attach LoRA adapters to the
        attention layers.  Drastically reduces trainable parameters.
    lora_r : int, optional
        LoRA rank (default 8).
    lora_alpha : int, optional
        LoRA scaling factor (default 16).
    lora_dropout : float, optional
        Dropout applied inside LoRA layers (default 0.05).
    torch_dtype : str, optional
        Torch dtype string for model weights, e.g. ``"float16"`` or
        ``"bfloat16"``.  Defaults to ``"float32"``.
    device_map : str or None, optional
        HuggingFace ``device_map`` argument.  Useful for multi-GPU / CPU
        offloading (e.g. ``"auto"``).
    **kwargs
        Additional keyword arguments forwarded to :class:`HuggingFaceModel`.

    Raises
    ------
    ImportError
        If ``transformers`` is not installed.
    ValueError
        If an unsupported ``task`` is provided.

    Examples
    --------
    Generation (default):

    >>> model = OLMoModel(task="generation", model_name="allenai/OLMo-1B")
    >>> smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    >>> predictions = model.predict(smiles)

    Classification:

    >>> model = OLMoModel(task="classification", n_tasks=1, n_classes=2)
    >>> model.fit(train_dataset, nb_epoch=3)

    Regression:

    >>> model = OLMoModel(task="regression", n_tasks=1)
    >>> model.fit(train_dataset, nb_epoch=5)
    """

    def __init__(
        self,
        task: str = "generation",
        model_name: str = _DEFAULT_MODEL,
        n_tasks: int = 1,
        n_classes: int = 2,
        tokenizer_path: Optional[str] = None,
        max_seq_length: int = _DEFAULT_MAX_LENGTH,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        torch_dtype: str = "float32",
        device_map: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers>=4.40.0 is required. "
                "Install via: pip install transformers"
            )

        task = task.lower()
        if task not in OLMO_SUPPORTED_TASKS:
            raise ValueError(
                f"task must be one of {OLMO_SUPPORTED_TASKS}, got '{task}'"
            )

        self.task = task
        self.model_name = model_name
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.max_seq_length = max_seq_length
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self._torch_dtype_str = torch_dtype
        self.device_map = device_map

        # Resolve torch dtype
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._torch_dtype = _dtype_map.get(torch_dtype, torch.float32)

        # Build tokenizer
        tok_path = tokenizer_path or model_name
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build backbone
        hf_model = self._build_model()

        # Optionally wrap with LoRA
        if use_lora:
            hf_model = self._apply_lora(hf_model)

        super().__init__(model=hf_model, task=task, tokenizer=self.tokenizer, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> "PreTrainedModel":
        """Instantiate the appropriate HuggingFace model for the chosen task."""
        common_kwargs: Dict[str, Any] = dict(
            pretrained_model_name_or_path=self.model_name,
            trust_remote_code=True,
            torch_dtype=self._torch_dtype,
        )
        if self.device_map is not None:
            common_kwargs["device_map"] = self.device_map

        if self.task == "generation":
            model = AutoModelForCausalLM.from_pretrained(**common_kwargs)

        elif self.task == "classification":
            config = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            config.num_labels = self.n_classes * self.n_tasks
            model = AutoModelForSequenceClassification.from_pretrained(
                **common_kwargs, config=config, ignore_mismatched_sizes=True
            )

        else:  # regression
            config = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            config.num_labels = 1  # single scalar per task
            # We load the causal LM and replace the lm_head
            base = AutoModelForCausalLM.from_pretrained(**common_kwargs)
            base.lm_head = RegressionHead(
                hidden_size=config.hidden_size,
                n_tasks=self.n_tasks,
            )
            model = base

        logger.info(
            "Loaded OLMo backbone: %s | task=%s | dtype=%s",
            self.model_name,
            self.task,
            self._torch_dtype_str,
        )
        return model

    def _apply_lora(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """Wrap *model* with LoRA adapters using the ``peft`` library."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            task_type_map = {
                "generation": TaskType.CAUSAL_LM,
                "classification": TaskType.SEQ_CLS,
                "regression": TaskType.SEQ_CLS,
            }
            lora_cfg = LoraConfig(
                task_type=task_type_map[self.task],
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            logger.info("LoRA adapters applied (r=%d, alpha=%d)", self.lora_r, self.lora_alpha)
        except ImportError:
            logger.warning(
                "peft not installed – LoRA disabled. "
                "Install via: pip install peft"
            )
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _tokenize(self, sequences: Sequence[str]) -> Dict[str, "torch.Tensor"]:
        """Tokenize a list of SMILES / text strings.

        Parameters
        ----------
        sequences : Sequence[str]
            Input strings to tokenize.

        Returns
        -------
        dict
            ``input_ids``, ``attention_mask`` tensors.
        """
        return self.tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

    def _pooled_hidden_state(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Extract the last-non-padding token hidden state for each sample.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, hidden_size)``.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]          # (B, L, H)
        # Grab the representation at the last real token
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, seq_lengths]         # (B, H)

    def generate_molecules(
        self,
        prompt: str = "",
        n_molecules: int = 5,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[str]:
        """Generate novel molecular strings via autoregressive sampling.

        Parameters
        ----------
        prompt : str, optional
            Conditioning prefix (e.g. a partial SMILES).
        n_molecules : int, optional
            Number of molecules to generate (default 5).
        max_new_tokens : int, optional
            Maximum number of tokens to generate per molecule.
        temperature : float, optional
            Sampling temperature.
        top_p : float, optional
            Nucleus sampling probability.
        do_sample : bool, optional
            If ``False``, use greedy decoding.

        Returns
        -------
        list of str
            Generated molecular strings (decoded, prompt stripped).
        """
        if self.task != "generation":
            raise RuntimeError(
                "generate_molecules is only available when task='generation'."
            )

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=n_molecules,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[-1]
        generated = [
            self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            for ids in output_ids
        ]
        return generated

    def save_lora_adapter(self, path: str) -> None:
        """Persist LoRA adapter weights to *path*.

        Parameters
        ----------
        path : str
            Directory where adapter weights will be saved.
        """
        if not self.use_lora:
            raise RuntimeError("LoRA was not enabled for this model.")
        self.model.save_pretrained(path)
        logger.info("LoRA adapter saved to %s", path)

    def load_lora_adapter(self, path: str) -> None:
        """Load LoRA adapter weights from *path*.

        Parameters
        ----------
        path : str
            Directory containing a previously saved LoRA adapter.
        """
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, path)
            logger.info("LoRA adapter loaded from %s", path)
        except ImportError:
            raise ImportError("peft is required to load LoRA adapters.")

    def get_num_trainable_params(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"OLMoModel(model_name={self.model_name!r}, task={self.task!r}, "
            f"n_tasks={self.n_tasks}, use_lora={self.use_lora})"
        )
