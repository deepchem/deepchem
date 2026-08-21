from typing import Optional, Dict, Union
import torch 
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM


class OLMoCausalLM(nn.Module):
    """
    Trainable OLMo causal language model for continued pretraining.

    This wraps HuggingFace AutoModelForCausalLM and relies on HuggingFace to compute the causal LM loss when labels are provided.
    """

    def __init__(
            self,
            model_name: str = "allenai/OLMo-7B-0724-hf",
            torch_dtype: torch.dtype = torch.float16,
            device_map: Union[str, Dict[str, int]] = "auto", 
    ) -> None:
        super().__init__()

        # Parameters are left unfrozen to support continued pretraining
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )


    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )