import torch.nn as nn
from transformers import AutoTokenizer, OlmoForCausalLM
from deepchem.models.torch_models.hf_models import HuggingFaceModel
try:
    import torch
except ModuleNotFoundError:
    pass


class Olmo(HuggingFaceModel):
    """Wrapper class for OLMo models for classification and regression tasks using a linear head
    Parameters
    ----------
    model : str
        Path to the pre-trained OLMo model
    tokenizer : AutoTokenizer
        Tokenizer for the OLMo model
    task_type : str
        Type of task to perform ('classification' or 'regression')
    n_tasks : int
        Number of tasks to predict
    Returns
    -------
        An instance of the Olmo model for classification or regression tasks
     Examples
     --------
     >>> from deepchem.models.torch_models.olmo_class import Olmo
     >>> model = Olmo(model="allenai/OLMo-1B-hf", tokenizer=None, task_type="classification", n_tasks=1)
     >>> dataset = dc.data.NumpyDataset(["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F", "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"], np.array([[1.0], [0.0]]))
     >>> loss = model.fit(dataset, nb_epoch=1)
     >>> predictions = model.predict(dataset)
    """

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
