import torch
import torch.nn as nn
from transformers import AutoTokenizer, OlmoForCausalLM
from deepchem.models.torch_models.hf_models import HuggingFaceModel


class OlmoForClassificationAndRegressionTasks(HuggingFaceModel):

    def __init__(self,
                 model,
                 tokenizer,
                 task_type: str = "classification",
                 n_tasks: int = 1,
                 **kwargs):
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'")

        if isinstance(model, str):
            model_path = model
            model = OlmoForCausalLM.from_pretrained(model_path,
                                                    torch_dtype=torch.bfloat16)
            model.gradient_checkpointing_enable()
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)

        model_dtype = next(model.parameters()).dtype
        model.score = nn.Linear(model.config.hidden_size,
                                n_tasks).to(dtype=model_dtype)

        original_forward = model.forward

        if task_type == "classification":

            def forward(*args, **fwd_kwargs):
                labels = fwd_kwargs.pop("labels", None)
                outputs = original_forward(*args,
                                           output_hidden_states=True,
                                           **fwd_kwargs)
                pooled = outputs.hidden_states[-1][:, -1, :]
                logits = model.score(pooled).float()
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(logits, labels.float())
                return {"loss": loss, "logits": logits}
        else:

            def forward(*args, **fwd_kwargs):
                labels = fwd_kwargs.pop("labels", None)
                outputs = original_forward(*args,
                                           output_hidden_states=True,
                                           **fwd_kwargs)
                pooled = outputs.hidden_states[-1][:, -1, :]
                logits = model.score(pooled).float()
                loss = None
                if labels is not None:
                    loss = nn.functional.mse_loss(logits, labels.float())
                return {"loss": loss, "logits": logits}

        model.forward = forward

        super().__init__(model=model,
                         tokenizer=tokenizer,
                         task=task_type,
                         **kwargs)
