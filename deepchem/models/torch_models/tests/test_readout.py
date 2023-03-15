import pytest

try:
    import torch
except ModuleNotFoundError:
    pass


@pytest.mark.torch
def testGroverReadout():
    from deepchem.models.torch_models.readout import GroverReadout
    n_nodes, n_features = 6, 32
    readout_mean = GroverReadout(rtype="mean")

    # testing a simple scenario where each embedding corresponds to an unique graph
    embedding = torch.ones(n_nodes, n_features)
    scope = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    readout = readout_mean(embedding, scope)
    assert readout.shape == (n_nodes, n_features)
    assert (readout == torch.ones(n_nodes, n_features)).all().tolist()

    # here embeddings 0, 1 belong to a scope, 2, 3 to another scope and 4, 5 to another scope
    # thus, we sill have 3 graphs
    n_graphs = n_nodes // 2
    scope = [(0, 2), (2, 2), (4, 2)]
    embedding[torch.tensor([0, 2, 4])] = torch.zeros_like(
        embedding[torch.tensor([0, 2, 4])])
    readout = readout_mean(embedding, scope)
    assert readout.shape == (n_graphs, n_features)
    assert (readout == torch.ones(n_graphs, n_features) / 2).all().tolist()

    attn_out = 8
    readout_attn = GroverReadout(rtype="self_attention",
                                 in_features=n_features,
                                 attn_hidden_size=32,
                                 attn_out_size=attn_out)

    readout = readout_attn(embedding, scope)
    assert readout.shape == (n_graphs, attn_out * n_features)
