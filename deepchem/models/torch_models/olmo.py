import gc
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModel, OlmoPreTrainedModel, OlmoForCausalLM, OlmoConfig, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from deepchem.models.torch_models.hf_models import HuggingFaceModel


class OlmoForSequenceClassification(OlmoPreTrainedModel):
    """OLMo model with a linear scoring head for classification or regression.

    Adds a linear layer on top of the last token's hidden state.

    Parameters
    ----------
    config : OlmoConfig
        Must have num_labels and problem_type set before constructing.

    Examples
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

    Loads the model architecture from a pretrained config in __init__ with
    randomly-initialised weights. Call load_from_pretrained(model_dir,
    from_hf_checkpoint=True) to replace those weights with a pretrained
    HuggingFace checkpoint.

    Classification/mtc uses BCEWithLogitsLoss with sigmoid outputs, regression/mtr
    uses MSELoss, and causal_lm uses OlmoForCausalLM. Gradient checkpointing is
    enabled for memory efficiency.

    Parameters
    ----------
    task_type : str, default 'classification'
        One of 'classification', 'regression', 'causal_lm', 'mtc', or 'mtr'.
    tokenizer_path : str, default 'allenai/OLMo-7B-hf'
        HuggingFace model ID or local path used to load the tokenizer and
        architecture config. Pretrained weights are not loaded here.
    n_tasks : int, default 1
        Number of output labels. Ignored for causal_lm.
    torch_dtype : str or torch.dtype or None, default None
        Dtype for model weights. Accepts a torch.dtype (e.g. torch.float16) or
        a string alias: 'float16'/'fp16', 'bfloat16'/'bf16', 'float32'/'fp32',
        'float64'/'fp64'. None uses the checkpoint's saved dtype.
    quantization_config : Optional[transformers.BitsAndBytesConfig], default None
        Quantization configuration used when loading pretrained weights via
        load_from_pretrained(..., from_hf_checkpoint=True). Has no effect on the
        randomly-initialised model built in __init__, since bitsandbytes
        quantization is only applied while a checkpoint's weights are being
        loaded through from_pretrained.
    **kwargs
        Forwarded to HuggingFaceModel.

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
                 finetune_strategy: str = "full_finetune",
                 **kwargs):
        if finetune_strategy not in ('lora', 'qlora', 'full_finetune'):
            raise ValueError(
                f"finetune_strategy must be 'lora', 'qlora', or 'full_finetune', "
                f"got '{finetune_strategy}'")
        self.n_tasks = n_tasks
        self.finetune_strategy = finetune_strategy
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

        self.model: Union[OlmoForCausalLM, OlmoForSequenceClassification]
        if task_type == "causal_lm":
            self.model = OlmoForCausalLM(olmo_config)
        else:
            if task_type in ("classification", "mtc"):
                problem_type = "multi_label_classification"
            else:
                problem_type = "regression"
            olmo_config.problem_type = problem_type
            olmo_config.num_labels = n_tasks
            self.model = OlmoForSequenceClassification(olmo_config)

        self.model.gradient_checkpointing_enable()  # type: ignore[union-attr]
        if isinstance(self._torch_dtype, torch.dtype):
            self.model = self.model.to(
                self._torch_dtype)  # type: ignore[union-attr]
        if self.finetune_strategy in ('lora', 'qlora'):
            self.model = self.apply_peft(self.model, task_type)

        super().__init__(
            model=self.model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            task=task_type,
            **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Tokenize inputs and cast labels to the model dtype for each task.

        Labels are cast to the model dtype to match the head output, as required by
        MSELoss and BCEWithLogitsLoss. causal_lm is used from the parent.

        Parameters
        ----------
        batch : tuple of (inputs, labels, weights)
            Raw batch where inputs[0] is a numpy array of SMILES strings,
            labels has shape (1, batch_size, n_tasks) or None during predict.

        Returns
        -------
        inputs : dict
            Tokenized inputs with a labels key, ready for model forward().
        y : torch.Tensor or None
            Label tensor on device, or None during prediction.
        w : torch.Tensor or None
            Sample weight tensor on device, or None.
        """

        smiles_batch, y, w = batch

        if w is not None:
            w = torch.tensor(w, dtype=torch.float).to(self.device)

        tokens = self.tokenizer(smiles_batch[0].tolist(),
                                padding=True,
                                return_tensors="pt")

        if self.task in ['regression', 'classification', 'mtr', 'mtc']:
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

    def build_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Build and return a BitsAndBytesConfig for qlora, or None for other strategies.

        Returns
        -------
        Optional[BitsAndBytesConfig]
            A 4-bit quantization config when finetune_strategy is 'qlora', else None.
        """
        if self.finetune_strategy == 'qlora':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        return None

    def apply_peft(self, model: PreTrainedModel,
                   task_type: str) -> PreTrainedModel:
        """Optionally apply k-bit training preparation and LoRA/QLoRA adapters to the model.

        Parameters
        ----------
        model: PreTrainedModel
            The loaded pretrained model.
        task_type: str
            DeepChem task type string, e.g. "causal_lm", "classification",
            "regression", "mtc", "mtr", or "multi_label_classification". Mapped
            internally to a PEFT TaskType ("CAUSAL_LM" for causal_lm, "SEQ_CLS"
            for everything else) since PEFT's LoraConfig only accepts its own
            enum values.

        Returns
        -------
        PreTrainedModel
            The model, possibly wrapped with PEFT adapters.
        """
        if self.finetune_strategy == 'qlora':
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True)
        if self.finetune_strategy in ('lora', 'qlora'):
            peft_task_type = TaskType.CAUSAL_LM if task_type == "causal_lm" else TaskType.SEQ_CLS
            lora_cfg = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=peft_task_type,
            )
            model = get_peft_model(model, lora_cfg)
        return model

    def load_from_pretrained(  # type: ignore[override]
            self,
            model_dir: Optional[str] = None,
            from_hf_checkpoint: bool = False):
        """Load pretrained OLMo weights into the current model instance.

        If from_hf_checkpoint is True, loads from a HuggingFace Hub model ID or a
        save_pretrained directory, replacing the randomly-initialised model built
        in __init__ with OlmoForCausalLM or OlmoForSequenceClassification
        depending on task. Quantization uses the constructor's
        quantization_config if supplied, else an auto-built 4-bit config when
        finetune_strategy == 'qlora'. apply_peft is then called to optionally
        wrap the model with LoRA/QLoRA adapters. If False, delegates to
        HuggingFaceModel.load_from_pretrained to load a local DeepChem checkpoint.

        Parameters
        ----------
        model_dir: str, optional
            HuggingFace Hub model ID, path to a save_pretrained directory, or path
            to a directory containing DeepChem checkpoints. Defaults to self.model_dir
            if not provided.
        from_hf_checkpoint: bool, default False
            When True, load from a HuggingFace Hub checkpoint using from_pretrained.
            When False, load from a local DeepChem checkpoint file.

        Example
        -------
        >>> # Loading from HuggingFace Hub and applying QLoRA adapters:
        >>> model = Olmo(task_type='regression', tokenizer_path='allenai/OLMo-7B-hf',
        ...              finetune_strategy='qlora')
        >>> model.load_from_pretrained('allenai/OLMo-7B-hf', from_hf_checkpoint=True)
        """
        if not from_hf_checkpoint:
            return super().load_from_pretrained(model_dir=model_dir,
                                                from_hf_checkpoint=False)

        if model_dir is None:
            model_dir = self.model_dir

        # init function creates a randomly initialised model. It is deleted before the
        # pretrained weights are loaded to reduce peak memory usage (having 2 copies of the model at the same time).
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        bnb_config = self._quantization_config or self.build_bnb_config()

        if self.task == "causal_lm":
            self.model = OlmoForCausalLM.from_pretrained(
                model_dir,
                quantization_config=bnb_config,
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
                quantization_config=bnb_config,
                torch_dtype=self._torch_dtype,
                low_cpu_mem_usage=True)

        self.model.gradient_checkpointing_enable()
        if bnb_config is None:
            # bitsandbytes-quantized models are placed on device during
            # from_pretrained and raise if .to() is called afterwards.
            self.model = self.model.to(self.device)

        if self.finetune_strategy in ('lora', 'qlora'):
            self.model = self.apply_peft(self.model, self.task)
