"""
Tests for EquivariantGNN and EquivariantGNNModel.
"""

import pytest
import numpy as np

try:
    import torch
    import dgl
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

try:
    from rdkit import Chem
    has_rdkit = True
except ModuleNotFoundError:
    has_rdkit = False

import deepchem as dc


@pytest.mark.torch
def test_equivariant_gnn_forward_conv():
    """Test forward pass of EquivariantGNN with convolution blocks."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    # Create a simple graph
    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata['x'] = torch.randn(3, 6)
    g.edata['edge_attr'] = torch.randn(6, 3)
    g.edata['w'] = torch.randn(6, 4)

    model = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='conv',
        pooling='avg'
    )

    out = model(g)
    assert out.shape == (1, 1)


@pytest.mark.torch
def test_equivariant_gnn_forward_attention():
    """Test forward pass of EquivariantGNN with attention blocks."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata['x'] = torch.randn(3, 6)
    g.edata['edge_attr'] = torch.randn(6, 3)
    g.edata['w'] = torch.randn(6, 4)

    model = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='attention',
        pooling='max',
        n_heads=2
    )

    out = model(g)
    assert out.shape == (1, 1)


@pytest.mark.torch
def test_equivariant_gnn_multi_task():
    """Test EquivariantGNN with multiple output tasks."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata['x'] = torch.randn(3, 6)
    g.edata['edge_attr'] = torch.randn(6, 3)
    g.edata['w'] = torch.randn(6, 4)

    model = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=5,
        num_degrees=3,
        edge_dim=4,
        block_type='conv'
    )

    out = model(g)
    assert out.shape == (1, 5)


@pytest.mark.torch
def test_equivariant_gnn_batched():
    """Test EquivariantGNN with batched graphs."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    # Create multiple graphs
    graphs = []
    for _ in range(3):
        g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
        g.ndata['x'] = torch.randn(3, 6)
        g.edata['edge_attr'] = torch.randn(6, 3)
        g.edata['w'] = torch.randn(6, 4)
        graphs.append(g)

    batched = dgl.batch(graphs)

    model = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='conv'
    )

    out = model(batched)
    assert out.shape == (3, 1)


@pytest.mark.torch
def test_equivariant_gnn_pooling_types():
    """Test different pooling strategies."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata['x'] = torch.randn(3, 6)
    g.edata['edge_attr'] = torch.randn(6, 3)
    g.edata['w'] = torch.randn(6, 4)

    for pooling in ['avg', 'max']:
        model = EquivariantGNN(
            num_layers=1,
            atom_feature_size=6,
            num_channels=8,
            n_tasks=1,
            num_degrees=3,
            edge_dim=4,
            block_type='conv',
            pooling=pooling
        )
        out = model(g)
        assert out.shape == (1, 1)


@pytest.mark.torch
def test_equivariant_gnn_gradient_flow():
    """Test that gradients flow through the model."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata['x'] = torch.randn(3, 6, requires_grad=True)
    g.edata['edge_attr'] = torch.randn(6, 3)
    g.edata['w'] = torch.randn(6, 4)

    model = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='conv'
    )

    out = model(g)
    loss = out.sum()
    loss.backward()

    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


@pytest.mark.torch
@pytest.mark.skipif(not has_rdkit, reason="RDKit not installed")
def test_equivariant_gnn_model_with_featurizer():
    """Test EquivariantGNNModel with EquivariantGraphFeaturizer."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNNModel

    smiles = ["CCO", "CC(=O)O", "C1=CC=CC=C1"]
    featurizer = dc.feat.EquivariantGraphFeaturizer(
        fully_connected=False, embeded=True
    )

    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        feat = featurizer.featurize(mol)
        if feat is not None and len(feat) > 0:
            mols.append(feat[0])

    if len(mols) == 0:
        pytest.skip("Featurization failed")

    labels = np.random.rand(len(mols), 1)
    weights = np.ones_like(labels)
    dataset = dc.data.NumpyDataset(X=mols, y=labels, w=weights)

    model = EquivariantGNNModel(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='conv',
        batch_size=2,
        device=torch.device('cpu')
    )

    # Test that fit runs without error
    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None


@pytest.mark.torch
@pytest.mark.skipif(not has_rdkit, reason="RDKit not installed")
def test_equivariant_gnn_model_attention():
    """Test EquivariantGNNModel with attention blocks."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNNModel

    smiles = ["CCO", "CCC"]
    featurizer = dc.feat.EquivariantGraphFeaturizer(
        fully_connected=False, embeded=True
    )

    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        feat = featurizer.featurize(mol)
        if feat is not None and len(feat) > 0:
            mols.append(feat[0])

    if len(mols) == 0:
        pytest.skip("Featurization failed")

    labels = np.random.rand(len(mols), 1)
    dataset = dc.data.NumpyDataset(X=mols, y=labels)

    model = EquivariantGNNModel(
        num_layers=1,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='attention',
        n_heads=2,
        batch_size=2,
        device=torch.device('cpu')
    )

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None


@pytest.mark.torch
def test_equivariant_gnn_with_normalization():
    """Test EquivariantGNN with and without normalization."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata['x'] = torch.randn(3, 6)
    g.edata['edge_attr'] = torch.randn(6, 3)
    g.edata['w'] = torch.randn(6, 4)

    # With normalization
    model_norm = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='conv',
        use_norm=True
    )

    # Without normalization
    model_no_norm = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='conv',
        use_norm=False
    )

    out_norm = model_norm(g)
    out_no_norm = model_no_norm(g)

    assert out_norm.shape == (1, 1)
    assert out_no_norm.shape == (1, 1)


@pytest.mark.torch
def test_equivariant_gnn_dropout():
    """Test EquivariantGNN with dropout."""
    from deepchem.models.torch_models.equivariant_gnn import EquivariantGNN

    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata['x'] = torch.randn(3, 6)
    g.edata['edge_attr'] = torch.randn(6, 3)
    g.edata['w'] = torch.randn(6, 4)

    model = EquivariantGNN(
        num_layers=2,
        atom_feature_size=6,
        num_channels=8,
        n_tasks=1,
        num_degrees=3,
        edge_dim=4,
        block_type='conv',
        dropout=0.5
    )

    # In training mode, dropout should be active
    model.train()
    out_train = model(g)

    # In eval mode, dropout should be inactive
    model.eval()
    out_eval = model(g)

    assert out_train.shape == (1, 1)
    assert out_eval.shape == (1, 1)
