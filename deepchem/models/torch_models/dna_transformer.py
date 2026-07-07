import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from deepchem.models.torch_models.torch_model import TorchModel


class DNATransformer(TorchModel):
    """
    DNA Transformer model wrapper using HuggingFace backend.

    This class wraps a HuggingFace transformer model and integrates
    it with DeepChem's TorchModel training infrastructure.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path.
    num_labels : int
        Number of output labels.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        **kwargs
    ):
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )

        super().__init__(
    model=hf_model,
    loss=nn.CrossEntropyLoss(),
    **kwargs
)
