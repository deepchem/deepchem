import os

import numpy as np

import deepchem as dc
import pytest


@pytest.mark.torch
def test_AtomEncoder():
    import torch

    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        full_atom_feature_dims,)
    from deepchem.models.torch_models.pna_gnn import AtomEncoder

    atom_encoder = AtomEncoder(emb_dim=32)

    num_samples = 10

    # Create input tensor with values within full_atom_feature_dims
    graph_features = torch.stack([
        torch.randint(low=0, high=dim, size=(num_samples,))
        for dim in full_atom_feature_dims
    ],
                                 dim=1)
    atom_embeddings = atom_encoder(graph_features)
    assert atom_embeddings.shape == (num_samples, 32)


@pytest.mark.torch
def test_BondEncoder():
    import torch

    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        full_bond_feature_dims,)
    from deepchem.models.torch_models.pna_gnn import BondEncoder

    bond_encoder = BondEncoder(emb_dim=32)

    num_samples = 10

    # Create input tensor with values within full_bond_feature_dims
    graph_features = torch.stack([
        torch.randint(low=0, high=dim, size=(num_samples,))
        for dim in full_bond_feature_dims
    ],
                                 dim=1)
    bond_embeddings = bond_encoder(graph_features)
    assert bond_embeddings.shape == (num_samples, 32)


@pytest.mark.torch
def test_pnalayer():
    import dgl
    import numpy as np
    import torch

    from deepchem.models.torch_models.pna_gnn import PNALayer
    in_dim = 32
    out_dim = 64
    in_dim_edges = 16
    aggregators = ["mean", "max"]
    scalers = ["identity", "amplification", "attenuation"]

    pna_layer = PNALayer(in_dim=in_dim,
                         out_dim=out_dim,
                         in_dim_edges=in_dim_edges,
                         aggregators=aggregators,
                         scalers=scalers)

    num_nodes = 10
    num_edges = 20
    node_features = torch.randn(num_nodes, in_dim)
    edge_features = torch.randn(num_edges, in_dim_edges)

    g = dgl.graph((np.random.randint(0, num_nodes, num_edges),
                   np.random.randint(0, num_nodes, num_edges)))
    g.ndata['feat'] = node_features
    g.edata['feat'] = edge_features

    g.ndata['feat'] = pna_layer(g)

    assert g.ndata['feat'].shape == (num_nodes, out_dim)


def get_regression_dataset():
    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        RDKitConformerFeaturizer,)
    np.random.seed(123)
    featurizer = RDKitConformerFeaturizer()
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_regression.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    return dataset, metric


@pytest.mark.torch
def testPNAGNN():
    import dgl
    from deepchem.models.torch_models.pna_gnn import PNAGNN

    data, _ = get_regression_dataset()
    features = dgl.batch([conformer.to_dgl_graph() for conformer in data.X])
    model = PNAGNN(hidden_dim=16,
                   aggregators=['mean', 'sum'],
                   scalers=['identity'])
    output = model(features)

    assert output.ndata['feat'].shape == (features.ndata['x'].shape[0], 16)


@pytest.mark.torch
def testPNA():
    import dgl
    from deepchem.models.torch_models.pna_gnn import PNA

    data, _ = get_regression_dataset()
    features = dgl.batch([conformer.to_dgl_graph() for conformer in data.X])
    target_dim = 1
    model = PNA(hidden_dim=16, target_dim=target_dim, task='regression')
    output = model(features)
    assert output.shape[1] == target_dim
