import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class OLMoBackbone(nn.Module):
    """
    Frozen HuggingFace OLMo backbone.
    Responsible only for producing hidden states.
    """

    def __init__(
            self,
            model_name: str = "allenai/OLMo-7B",
            torch_dtype=torch.float16,
            device_map = "auto",
    ):
        super().__init__()


        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        for p in self.model.parameters():
            p.requires_grad = False

    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1]
    

class OLMoPropertyModel(nn.Module):
    """
    OLMo backbone + lightweight regression head.
    Designed for molecular property prediction.
    """


    def __init__(self, backbone: OLMoBackbone, hidden_dim: int = 4096):
        super().__init__()
        self.backbone = backbone

        self.head = nn.Linear(hidden_dim, 1).to(
            device=next(backbone.parameters()).device,
            dtype=next(backbone.parameters()).dtype,
        )

    def forward(self , input_ids, attention_mask=None):
        hidden = self.backbone(input_ids, attention_mask)
        pooled = hidden.mean(dim=1)
        return self.head(pooled)