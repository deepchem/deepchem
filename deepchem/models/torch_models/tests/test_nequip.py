import pytest
import torch

from deepchem.models.torch_models import NequIP

try:
    import dgl
    has_dgl = True
except ImportError:
    has_dgl = False


@pytest.mark.torch
def test_nequip_forward():
    """Test NequIP forward pass."""
    if not has_dgl:
        pytest.skip('DGL is not installed')

    # 1. Create Model
    model = NequIP(num_layers=2, max_ell=2, num_features=2, hidden_dim=4)

    # 2. Create Dummy Graph
    # 2 nodes, connected
    g = dgl.graph(([0, 1], [1, 0]))
    # H, C
    g.ndata['x'] = torch.tensor([[1], [6]], dtype=torch.float32)
    g.ndata['pos'] = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                                  dtype=torch.float32,
                                  requires_grad=True)

    src, dst = g.edges()
    g.edata['edge_features'] = g.ndata['pos'][dst] - g.ndata['pos'][src]

    # 3. Forward
    z = g.ndata['x'].long().squeeze()
    e, f = model(g, z)

    # Check shapes
    # (Batch, 1)
    assert e.shape == (1, 1)
    # (Nodes, 3)
    assert f.shape == (2, 3)

    # Check grad
    assert f.requires_grad or f.grad_fn is not None
