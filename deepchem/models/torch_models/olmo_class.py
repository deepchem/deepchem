from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from deepchem.data import Dataset

class OlmoClass(HuggingFaceModel):
    MODEL_NAME = "allenai/OLMo-1B-hf"

    def __init__(self, model_dir=None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME,
                                                  trust_remote_code=True)
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16
            if torch.cuda.is_available() else torch.float32,
        )

        super(OlmoClass, self).__init__(
            model=model,
            tokenizer=tokenizer,
            task="causal_lm",
            model_dir=model_dir,
            **kwargs)

    #def generate(self, dataset: Dataset, **kwargs):  # Generate Function already in hf_models.py
        #return super(OlmoClass, self).generate(dataset, **kwargs) #Commented it out since it is already in hf_models.py

    def regression(self, dataset: Dataset, n_tasks: int = 1, **kwargs):
        if not hasattr(self.model, 'original_forward'):
            self.model.original_forward = self.model.forward
        model_dtype = next(self.model.parameters()).dtype
        self.model.score = nn.Linear(self.model.config.hidden_size, n_tasks).to(device=self.device, dtype=model_dtype)
        self.score = self.model.score
        original_forward = self.model.original_forward

        def forward(*args, **kwargs):
            labels = kwargs.pop("labels", None)
            outputs = original_forward(*args, output_hidden_states=True, **kwargs)
            hidden = outputs.hidden_states[-1]
            pooled = hidden[:, -1, :]
            logits = self.score(pooled)
            loss = None
            if labels is not None:
                loss = nn.functional.mse_loss(logits.float(), labels.float())
            return {"loss": loss, "logits": logits}
        self.model.forward = forward
        self.task = 'regression'
        return self.fit(dataset, **kwargs)

    def classification(self, dataset: Dataset, n_tasks: int = 2, **kwargs):
        if not hasattr(self.model, 'original_forward'):
            self.model.original_forward = self.model.forward
        model_dtype = next(self.model.parameters()).dtype
        self.model.score = nn.Linear(self.model.config.hidden_size, n_tasks).to(device=self.device, dtype=model_dtype)
        self.score = self.model.score
        original_forward = self.model.original_forward

        def forward(*args, **kwargs):
            labels = kwargs.pop("labels", None)
            outputs = original_forward(*args, output_hidden_states=True, **kwargs)
            hidden = outputs.hidden_states[-1]
            pooled = hidden[:, -1, :]
            logits = self.score(pooled)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits.float(), labels.float())
            return {"loss": loss, "logits": logits}
        self.model.forward = forward
        self.task = 'classification'
        return self.fit(dataset, **kwargs)