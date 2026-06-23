import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM
from deepchem.models.torch_models.torch_model import TorchModel


class DNATransformerForMLM(TorchModel):
    """
    DNA Transformer for Masked Language Modeling.
    Uses HuggingFace AutoModelForMaskedLM backend.
    """

    def __init__(self, model_name="bert-base-uncased", **kwargs):

        config = AutoConfig.from_pretrained(model_name)
        hf_model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)

        super().__init__(
            model=hf_model,
            loss=nn.CrossEntropyLoss(ignore_index=-100),
            **kwargs
        )
