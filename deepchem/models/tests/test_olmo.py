import torch
import torch.nn as nn
import numpy as np
import deepchem as dc

from deepchem.models.torch_models.olmo import OLMoPropertyModel


class DummyBackbone(nn.Module):
    """Lightweight mock backbone for unit testing."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask = None):
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.hidden_dim, requires_grad=True)
    

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


def test_olmo_overfits():
    """Test that OLMoPropertyModel can overfit a tiny dataset."""
    torch.manual_seed(0)
    np.random.seed(0)

    # Tiny fake dataset
    X = np.random.randint(0, 10, size=(4, 5)).astype(np.int64)
    y = np.ones((4, 1), dtype = np.float32)

    dataset = dc.data.NumpyDataset(X, y)

    hidden_dim = 32
    backbone = DummyBackbone(hidden_dim=hidden_dim)
    model = OLMoPropertyModel(backbone = backbone, hidden_dim = hidden_dim)

    dc_model = dc.models.TorchModel(
        model=model,
        loss=dc.models.losses.L2Loss(),
        batch_size=2,
        learning_rate=0.05,
    )

    final_loss = dc_model.fit(dataset, nb_epoch=30)

    assert final_loss < 0.1