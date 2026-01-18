import torch
import torch.nn as nn

from deepchem.models.torch_models.olmo import OLMoPropertyModel


class DummyBackbone(nn.Module):
    """Lightweight mock backbone for unit testing."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask = None):
        bacth_size, seq_len = input_ids.shape
        return torch.randn(bacth_size, seq_len, self.hidden_dim)
    

def test_olmo_property_model_forward():
    """Test forward pass of OLMoPropertyModel with a dummy backbone."""
    batch_size = 4
    seq_len = 10
    hidden_dim = 16


    backbone = DummyBackbone(hidden_dim=hidden_dim)
    model = OLMoPropertyModel(backbone=backbone, hidden_dim=hidden_dim)


    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)


    outputs = model(input_ids, attention_mask)


    assert outputs.shape == (batch_size, 1)