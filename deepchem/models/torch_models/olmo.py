import gc
from contextlib import nullcontext
import torch
import torch.nn as nn
from typing import Any, ContextManager, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel, OlmoPreTrainedModel, OlmoForCausalLM, OlmoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights  # type: ignore[no-redef]
from deepchem.models.torch_models.hf_models import HuggingFaceModel


class OlmoForSequenceClassification(OlmoPreTrainedModel):
    """OLMo with a linear scoring head over the last token's hidden state.

    Parameters
    ----------
    config : OlmoConfig
        Must have num_labels and problem_type set.

    Example
    --------
    >>> model = OlmoForSequenceClassification.from_pretrained(
    ...     "allenai/OLMo-7B-hf", num_labels=1,
    ...     problem_type="multi_label_classification")
    """

    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = AutoModel.from_config(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.score = self.score.to(next(self.parameters()).dtype)
        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                **kwargs):
        """Forward pass for OLMo sequence classification/regression."""
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        batch_size = (input_ids.shape[0]
                      if input_ids is not None else hidden_states.shape[0])

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) -
                    1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device),
                               sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type == "regression":
                loss = nn.MSELoss()(pooled_logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss = nn.BCEWithLogitsLoss()(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=getattr(outputs, 'past_key_values', None),
            hidden_states=getattr(outputs, 'hidden_states', None),
            attentions=getattr(outputs, 'attentions', None),
        )


DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
    "auto": "auto"
}


class Olmo(HuggingFaceModel):
    """OLMo wrapper for classification, regression, and causal language modelling.

    __init__ builds the architecture with random weights. Call
    load_from_pretrained(model_dir, from_hf_checkpoint=True) to load a
    pretrained HuggingFace checkpoint.

    Parameters
    ----------
    task_type : str, default 'classification'
        One of 'classification', 'regression', 'causal_lm', 'mtc', or 'mtr'.
    tokenizer_path : str, default 'allenai/OLMo-7B-hf'
        HuggingFace model ID or local path for the tokenizer and config.
    n_tasks : int, default 1
        Number of output labels. Ignored for causal_lm.
    torch_dtype : str or torch.dtype or None, default None
        Dtype for model weights. String aliases: 'float16'/'fp16',
        'bfloat16'/'bf16', 'float32'/'fp32', 'float64'/'fp64'.
    quantization_config : Optional[transformers.BitsAndBytesConfig], default None
        Used only by load_from_pretrained(..., from_hf_checkpoint=True).
    gradient_checkpointing : bool, default False
        Trade compute for memory by recomputing activations in backward.
    skip_weight_init : bool, default False
        Skip random weight init in __init__ for faster construction of large
        models.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from deepchem.models.torch_models.olmo import Olmo
    >>> model = Olmo(task_type="classification", tokenizer_path="allenai/OLMo-7B-hf",
    ...              n_tasks=1)
    >>> model.load_from_pretrained("allenai/OLMo-7B-hf", from_hf_checkpoint=True)
    >>> dataset = dc.data.NumpyDataset(
    ...     ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    ...      "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"],
    ...     np.array([[1.0], [0.0]]))
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> predictions = model.predict(dataset)
    """

    def __init__(self,
                 task_type: str = "classification",
                 tokenizer_path: str = "allenai/OLMo-7B-hf",
                 n_tasks: int = 1,
                 torch_dtype=None,
                 quantization_config=None,
                 gradient_checkpointing: bool = False,
                 skip_weight_init: bool = False,
                 **kwargs):
        self.n_tasks = n_tasks
        self._gradient_checkpointing = gradient_checkpointing
        if isinstance(torch_dtype, str):
            if torch_dtype not in DTYPES:
                raise ValueError(
                    f"torch_dtype '{torch_dtype}' is not recognized. "
                    f"Valid string values are: {list(DTYPES.keys())}")
            torch_dtype = DTYPES[torch_dtype]
        self._torch_dtype = torch_dtype
        self._quantization_config = quantization_config

        if task_type not in ("classification", "regression", "causal_lm", "mtc",
                             "mtr"):
            raise ValueError(
                f"task_type must be 'classification', 'regression', 'causal_lm', 'mtc', or 'mtr', "
                f"got '{task_type}'")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        olmo_config = OlmoConfig.from_pretrained(tokenizer_path)

        model: Union[OlmoForCausalLM, OlmoForSequenceClassification]
        init_ctx: ContextManager
        if skip_weight_init:
            init_ctx = no_init_weights()
        else:
            init_ctx = nullcontext()
        with init_ctx:
            if task_type == "causal_lm":
                model = OlmoForCausalLM(olmo_config)
            else:
                if task_type in ("classification", "mtc"):
                    problem_type = "multi_label_classification"
                else:
                    problem_type = "regression"
                olmo_config.problem_type = problem_type
                olmo_config.num_labels = n_tasks
                model = OlmoForSequenceClassification(olmo_config)

        if skip_weight_init:
            model.tie_weights()  # type: ignore[union-attr]

        if self._gradient_checkpointing:
            model.gradient_checkpointing_enable()  # type: ignore[union-attr]
        if isinstance(self._torch_dtype, torch.dtype):
            model = model.to(self._torch_dtype)  # type: ignore[union-attr]

        super().__init__(
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            task=task_type,
            **kwargs)

    def load_from_pretrained(  # type: ignore[override]
            self,
            model_dir: Optional[str] = None,
            from_hf_checkpoint: bool = False):
        """Load OLMo weights into the current model instance.

        Parameters
        ----------
        model_dir : str, optional
            HuggingFace Hub model ID, save_pretrained directory, or DeepChem
            checkpoint directory. Defaults to self.model_dir.
        from_hf_checkpoint : bool, default False
            True loads a HuggingFace checkpoint via from_pretrained. False
            loads a local DeepChem checkpoint.
        """
        if not from_hf_checkpoint:
            return super().load_from_pretrained(model_dir=model_dir,
                                                from_hf_checkpoint=False)

        if model_dir is None:
            model_dir = self.model_dir

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.task == "causal_lm":
            self.model = OlmoForCausalLM.from_pretrained(
                model_dir,
                quantization_config=self._quantization_config,
                torch_dtype=self._torch_dtype,
                low_cpu_mem_usage=True)
        else:
            if self.task in ("classification", "mtc"):
                problem_type = "multi_label_classification"
            else:
                problem_type = "regression"
            self.model = OlmoForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=self.n_tasks,
                problem_type=problem_type,
                quantization_config=self._quantization_config,
                torch_dtype=self._torch_dtype,
                low_cpu_mem_usage=True)

        if self._gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self._quantization_config is None:
            # bitsandbytes-quantized models are placed on device during
            # from_pretrained and raise if .to() is called afterwards.
            self.model = self.model.to(self.device)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Tokenize inputs and cast labels to the model dtype. causal_lm uses the parent.

        Parameters
        ----------
        batch : tuple of (inputs, labels, weights)
            inputs[0] is a numpy array of SMILES strings; labels has shape
            (1, batch_size, n_tasks) or None during predict.

        Returns
        -------
        inputs : dict
            Tokenized inputs with a labels key.
        y : torch.Tensor or None
            Label tensor on device, or None during prediction.
        w : torch.Tensor or None
            Sample weight tensor on device, or None.
        """

        smiles_batch, y, w = batch

        if w is not None:
            w = torch.tensor(w, dtype=torch.float).to(self.device)

        if self.task in ['regression', 'classification', 'mtr', 'mtc']:
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "right"
            tokens = self.tokenizer(smiles_batch[0].tolist(),
                                    padding=True,
                                    return_tensors="pt")
            self.tokenizer.padding_side = original_padding_side

            if y is not None:
                # y is None during predict
                model_dtype = next(self.model.parameters()).dtype
                y = torch.from_numpy(y[0]).to(dtype=model_dtype,
                                              device=self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, 'labels': y}
            return inputs, y, w
        else:
            return super()._prepare_batch(batch)
