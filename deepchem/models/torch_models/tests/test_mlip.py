import pytest
import numpy as np
import torch
import deepchem as dc
from deepchem.models.torch_models import MLIPModel

try:
    import dgl
    has_dgl = True
except ImportError:
    has_dgl = False


@pytest.mark.torch
def test_mlip_init():
    """Test MLIPModel initialization."""
    if not has_dgl:
        pytest.skip('DGL is not installed')

    class DummyBackbone(torch.nn.Module):

        def forward(self, g, z):
            return torch.tensor([[1.0]]), torch.tensor([[0.0, 0.0, 0.0]])

    backbone = DummyBackbone()
    model = MLIPModel(backbone)
    assert model.energy_weight == 1.0
    assert model.force_weight == 100.0


@pytest.mark.torch
def test_overfit():
    """Test overfitting on a small dataset."""
    if not has_dgl:
        pytest.skip('DGL is not installed')

    # 1. Define Dummy Backbone
    class DummyNequIP(torch.nn.Module):

        def __init__(self):
            super(DummyNequIP, self).__init__()
            self.embedding = torch.nn.Embedding(10, 8)
            self.out = torch.nn.Linear(8, 1)

        def forward(self, g, z):
            # Simple energy prediction from node features
            h = self.embedding(z)
            g.ndata['h'] = h
            # Sum node features to get graph feature
            h_g = dgl.sum_nodes(g, 'h')
            energy = self.out(h_g)
            # Dummy forces (gradient of energy w.r.t pos would be better, but for dummy test...)
            forces = torch.zeros((g.num_nodes(), 3))
            return energy, forces

    # 2. Create Dummy Dataset
    # Graph 1
    g1 = dgl.graph(([0], [1]))
    # Atomic numbers as float
    g1.ndata['x'] = torch.tensor([[1], [1]], dtype=torch.float32)
    g1.ndata['pos'] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                   dtype=torch.float32)

    # Graph 2
    g2 = dgl.graph(([0], [1]))
    g2.ndata['x'] = torch.tensor([[2], [2]], dtype=torch.float32)
    g2.ndata['pos'] = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                                   dtype=torch.float32)

    # Labels: Energy (1) + Forces (N, 3)
    y1 = [1.0, np.zeros((2, 3))]
    y2 = [2.0, np.zeros((2, 3))]

    # Wrap in DeepChem GraphData/NumpyDataset
    from deepchem.feat.graph_data import GraphData

    def to_graph_data(g):
        return GraphData(node_features=g.ndata['x'].numpy(),
                         edge_index=np.stack(g.edges()),
                         node_pos_features=g.ndata['pos'].numpy())

    ds = dc.data.NumpyDataset(X=[to_graph_data(g1),
                                 to_graph_data(g2)],
                              y=[y1, y2])

    # 3. Train
    backbone = DummyNequIP()
    model = MLIPModel(backbone, batch_size=2, learning_rate=0.01)

    # Increased epochs for convergence
    model.fit(ds, nb_epoch=50)

    # 4. Predict
    preds = model.predict(ds)
    # preds is list of (E, F)

    # Energy error
    e_preds = preds[0]
    assert np.allclose(e_preds, [1.0, 2.0], atol=0.5)
