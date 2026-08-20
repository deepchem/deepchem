import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    import torch
    has_torch = True
    from deepchem.models.torch_models.ProteinMPNN import (
        PositionalEncodings,
        ProteinFeaturesLayer,
    )
    from deepchem.feat.ProteinMPNN_featurizer import (
        ProteinStructureData,
        _MapperProteinMPNN,
    )
except ImportError:
    has_torch = False


@pytest.mark.torch
def test_positional_encodings_init():
    """Test that PositionalEncodings initializes its linear layer correctly."""
    num_embeddings = 8
    max_relative_feature = 4

    layer = PositionalEncodings(num_embeddings=num_embeddings,
                                max_relative_feature=max_relative_feature)

    assert layer.num_embeddings == num_embeddings
    assert layer.max_relative_feature == max_relative_feature
    assert layer.linear.in_features == 2 * max_relative_feature + 2
    assert layer.linear.out_features == num_embeddings


@pytest.mark.torch
def test_positional_encodings_output_shape():
    """Test that PositionalEncodings returns embeddings with the correct shape.

    Given offset and mask tensors of shape ``(batch, num_nodes, k)``, the
    output should have shape ``(batch, num_nodes, k, num_embeddings)``.
    """
    batch, num_nodes, k = 2, 6, 4
    num_embeddings = 16

    layer = PositionalEncodings(num_embeddings=num_embeddings)
    offset = torch.randint(-5, 6, (batch, num_nodes, k))
    mask = torch.ones(batch, num_nodes, k)

    out = layer(offset, mask)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([batch, num_nodes, k, num_embeddings])


@pytest.mark.torch
def test_protein_features_layer_init():
    """Test that ProteinFeaturesLayer initializes its submodules correctly."""
    edge_features = 64
    num_positional_embeddings = 8
    num_rbf = 4
    top_k = 10
    augment_eps = 0.1

    layer = ProteinFeaturesLayer(
        edge_features=edge_features,
        num_positional_embeddings=num_positional_embeddings,
        num_rbf=num_rbf,
        top_k=top_k,
        augment_eps=augment_eps)

    assert layer.edge_features == edge_features
    assert layer.top_k == top_k
    assert layer.augment_eps == augment_eps
    assert layer.num_rbf == num_rbf
    assert isinstance(layer.embeddings, PositionalEncodings)
    assert layer.edge_embedding.in_features == num_positional_embeddings + num_rbf * 25
    assert layer.edge_embedding.out_features == edge_features


@pytest.mark.torch
def test_protein_features_layer_dist_clips_top_k():
    """Test that dist() clips k when the number of residues is less than top_k."""
    layer = ProteinFeaturesLayer(edge_features=32, top_k=10)
    num_nodes = 4
    Ca = torch.rand(1, num_nodes, 3)
    mask = torch.ones(1, num_nodes)

    D_neighbors, E_idx = layer.dist(Ca, mask)

    assert D_neighbors.shape == torch.Size([1, num_nodes, num_nodes])
    assert E_idx.shape == torch.Size([1, num_nodes, num_nodes])


@pytest.mark.torch
def test_protein_features_layer_dist_neighbors():
    """Test that dist() selects spatially close residues as neighbors.

    Constructs residues on a line and verifies that a query residue prefers
    nearby indices over a distant one.
    """
    layer = ProteinFeaturesLayer(edge_features=32, top_k=3)
    Ca = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [10., 0.,
                                                                   0.]]])
    mask = torch.ones(1, 4)

    _, E_idx = layer.dist(Ca, mask)

    assert 1 in E_idx[0, 0].tolist()
    assert 2 in E_idx[0, 0].tolist()
    assert 3 not in E_idx[0, 0].tolist()


@pytest.mark.torch
def test_protein_features_layer_dist_masked_residues():
    """Test that dist() excludes masked residues from neighbor selection.

    Verifies that masked residues are not selected as neighbors for valid
    query residues.
    """
    layer = ProteinFeaturesLayer(edge_features=32, top_k=2)
    Ca = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [5., 0., 0.], [6., 0.,
                                                                   0.]]])
    mask = torch.tensor([[1., 0., 1., 1.]])

    _, E_idx = layer.dist(Ca, mask)

    for node in [0, 2, 3]:
        assert 1 not in E_idx[0, node].tolist()


@pytest.mark.torch
def test_protein_features_layer_rbf_peak_near_center():
    """Test that rbf() assigns the largest response near the matching center."""
    layer = ProteinFeaturesLayer(edge_features=32, num_rbf=4, top_k=1)
    D = torch.tensor([[[[2.0]]]])

    out = layer.rbf(D)

    assert out.argmax(dim=-1).item() == 0


@pytest.mark.torch
def test_protein_features_layer_get_rbf_output_shape():
    """Test that get_rbf() returns gathered pairwise RBF features."""
    layer = ProteinFeaturesLayer(edge_features=32, num_rbf=8, top_k=3)
    A = torch.rand(1, 5, 3)
    B = torch.rand(1, 5, 3)
    E_idx = torch.randint(0, 5, (1, 5, 3))

    out = layer.get_rbf(A, B, E_idx)

    assert out.shape == torch.Size([1, 5, 3, 8])


@pytest.mark.torch
def test_protein_features_layer_get_rbf_values():
    """Test that get_rbf() matches rbf() on manually gathered distances."""
    layer = ProteinFeaturesLayer(edge_features=32, num_rbf=4, top_k=2)
    A = torch.tensor([[[0., 0., 0.], [3., 0., 0.], [6., 0., 0.]]])
    B = torch.tensor([[[0., 0., 0.], [4., 0., 0.], [6., 0., 0.]]])
    E_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])

    out = layer.get_rbf(A, B, E_idx)

    # Node 0, neighbor 1: distance between A[0] and B[1] is 4.0
    assert torch.allclose(out[0, 0, 0],
                          layer.rbf(torch.tensor([[[4.0]]]))[0, 0, 0])
    # Node 0, neighbor 2: distance between A[0] and B[2] is 6.0
    assert torch.allclose(out[0, 0, 1],
                          layer.rbf(torch.tensor([[[6.0]]]))[0, 0, 0])


@pytest.mark.torch
def test_protein_features_layer_forward_output_shape():
    """Test that forward() returns edge features and neighbor indices."""
    batch, num_nodes = 2, 12
    edge_features = 64
    top_k = 5

    layer = ProteinFeaturesLayer(edge_features=edge_features, top_k=top_k)
    X, mask, residue_idx, chain_labels = _make_structure_inputs(
        batch, num_nodes)

    E, E_idx = layer(X, mask, residue_idx, chain_labels)

    assert E.shape == torch.Size([batch, num_nodes, top_k, edge_features])
    assert E_idx.shape == torch.Size([batch, num_nodes, top_k])
    assert torch.isfinite(E).all()


@pytest.mark.torch
def test_protein_features_layer_forward_batched():
    """Test that forward() handles independent batch elements correctly."""
    batch, num_nodes = 3, 8
    layer = ProteinFeaturesLayer(edge_features=32, top_k=4)

    X, mask, residue_idx, chain_labels = _make_structure_inputs(
        batch, num_nodes)
    E, E_idx = layer(X, mask, residue_idx, chain_labels)

    assert E.shape[0] == batch
    assert E_idx.shape[0] == batch


@pytest.mark.torch
def test_protein_features_layer_forward_augmentation():
    """Test that coordinate augmentation is applied only during training."""
    torch.manual_seed(0)
    layer = ProteinFeaturesLayer(edge_features=32, top_k=4, augment_eps=0.5)
    X, mask, residue_idx, chain_labels = _make_structure_inputs(1, 10)

    layer.eval()
    E_eval_1, _ = layer(X, mask, residue_idx, chain_labels)
    E_eval_2, _ = layer(X, mask, residue_idx, chain_labels)
    assert torch.allclose(E_eval_1, E_eval_2)

    layer.train()
    E_train_1, _ = layer(X, mask, residue_idx, chain_labels)
    E_train_2, _ = layer(X, mask, residue_idx, chain_labels)
    assert not torch.allclose(E_train_1, E_train_2)


@pytest.mark.torch
def test_protein_features_layer_forward_with_featurizer_output():
    """Test forward() with tensors derived from the ProteinMPNN featurizer."""
    coords = np.array(
        [
            [[0.000, 0.000, 0.000], [1.458, 0.000, 0.000],
             [2.009, 1.424, 0.000], [1.319, 2.441, 0.000]],
            [[3.326, 1.488, 0.000], [4.015, 2.766, 0.000],
             [5.516, 2.517, 0.000], [6.155, 3.421, 0.000]],
        ],
        dtype=np.float32,
    )
    structure = ProteinStructureData(backbone_coords=coords, sequence='AG')
    mapper = _MapperProteinMPNN(structure)
    X_np, _, mask_np, _, residue_idx_np, chain_encoding_np = mapper.values

    X = torch.from_numpy(X_np).float().unsqueeze(0)
    mask = torch.from_numpy(mask_np).float().unsqueeze(0)
    residue_idx = torch.from_numpy(residue_idx_np).unsqueeze(0)
    chain_labels = torch.from_numpy(chain_encoding_np).float().unsqueeze(0)

    layer = ProteinFeaturesLayer(edge_features=128, top_k=2)
    E, E_idx = layer(X, mask, residue_idx, chain_labels)

    assert E.shape == torch.Size([1, 2, 2, 128])
    assert E_idx.shape == torch.Size([1, 2, 2])
    assert torch.isfinite(E).all()


def _make_structure_inputs(batch: int, num_nodes: int):
    """Helper to create random protein structure tensors for model tests."""
    X = torch.randn(batch, num_nodes, 4, 3)
    mask = torch.ones(batch, num_nodes)
    residue_idx = torch.arange(num_nodes).unsqueeze(0).expand(batch, -1)
    chain_labels = torch.ones(batch, num_nodes)
    return X, mask, residue_idx, chain_labels
