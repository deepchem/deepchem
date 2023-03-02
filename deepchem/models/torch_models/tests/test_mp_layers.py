import pytest


@pytest.mark.torch
def gen_dataset():
    x, edge_index, edge_attr, y = gen_data()

@pytest.mark.torch
def test_GINConv():
    from deepchem.models.torch_models.mp_layers import GINConv
    layer = GINConv(emb_dim=10)


@pytest.mark.torch
def test_GCNConv():
    from deepchem.models.torch_models.mp_layers import GCNConv
    layer = GCNConv(emb_dim=10)


@pytest.mark.torch
def test_GATConv():
    from deepchem.models.torch_models.mp_layers import GATConv
    layer = GATConv(emb_dim=10)


@pytest.mark.torch
def test_GraphSAGEConv():
    from deepchem.models.torch_models.mp_layers import GraphSAGEConv
    layer = GraphSAGEConv(emb_dim=10)
