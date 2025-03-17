import pytest
import numpy as np
import deepchem as dc

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_se3_layer_norm():
    """Test SE3LayerNorm layer."""
    from deepchem.models.torch_models.layers import SE3LayerNorm
    batch_size, num_channels = 10, 30
    layer = SE3LayerNorm(num_channels)
    x = torch.randn(batch_size, num_channels)
    output = layer(x)

    assert output.shape == x.shape
    assert torch.allclose(output.mean(dim=-1),
                          torch.zeros(batch_size),
                          atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize("num_freq, in_dim, out_dim, edge_dim", [(5, 10, 15, 3),
                                                                 (2, 8, 16, 2)])
def test_se3radial_func(num_freq, in_dim, out_dim, edge_dim):
    from deepchem.models.torch_models.layers import SE3RadialFunc
    layer = SE3RadialFunc(num_freq, in_dim, out_dim, edge_dim)
    x = torch.randn(8, edge_dim + 1)
    output = layer(x)

    assert output.shape == (8, out_dim, 1, in_dim, 1, num_freq)


def rotation_matrix(axis, angle):
    """Generate a 3D rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([[
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)
    ], [
        2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)
    ], [
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c
    ]])


def apply_rotation(x, axis, angle):
    """Apply a 3D rotation to the positions."""
    R = rotation_matrix(axis, angle)
    return torch.tensor(np.dot(x.numpy(), R), dtype=torch.float32)


@pytest.mark.torch
@pytest.mark.parametrize("max_degree, nc_in, nc_out, edge_dim",
                         [(3, 32, 128, 5)])
def test_se3pairwiseconv_equivariance(max_degree, nc_in, nc_out, edge_dim):
    """Test SE(3) equivariance of SE3PairwiseConv using a real molecular graph (CCO)."""
    from rdkit import Chem
    import dgl
    from deepchem.models.torch_models.layers import SE3PairwiseConv
    from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r

    # Load molecule and featurize
    mol = Chem.MolFromSmiles("CCO")
    featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True,
                                                    embeded=True)
    mol_graph = featurizer.featurize([mol])[0]

    # Initialize SE3PairwiseConv layer
    pairwise_conv = SE3PairwiseConv(degree_in=0,
                                    nc_in=32,
                                    degree_out=0,
                                    nc_out=128,
                                    edge_dim=5)

    G = dgl.graph((mol_graph.edge_index[0], mol_graph.edge_index[1]))
    G.ndata['f'] = torch.tensor(mol_graph.node_features,
                                dtype=torch.float32).unsqueeze(-1)
    G.ndata['x'] = torch.tensor(mol_graph.positions,
                                dtype=torch.float32)  # Atomic positions
    G.edata['d'] = torch.tensor(mol_graph.edge_features, dtype=torch.float32)
    G.edata['w'] = torch.tensor(mol_graph.edge_weights, dtype=torch.float32)

    # Compute initial SE(3) equivariant basis and r
    basis, r = get_equivariant_basis_and_r(G, max_degree)

    # Compute radial distances and edge features before rotation
    # r = torch.sqrt(torch.sum(G.edata["d"]**2, -1, keepdim=True))
    feat = torch.cat([G.edata["w"], r], -1) if "w" in G.edata else torch.cat(
        [r], -1)

    # Apply random graph rotation to nodes
    axis = np.array([1.0, 1.0, 1.0])  # Rotate around (1,1,1) axis
    angle = np.pi / 4  # 45-degree rotation
    G.ndata['x'] = apply_rotation(G.ndata['x'], axis, angle)  # Apply rotation

    # Compute SE(3) rotated equivariant basis and r
    basis_rotated, r_rotated = get_equivariant_basis_and_r(G, max_degree)

    # PairwiseConv forward pass for original graph
    output_original = pairwise_conv(feat, basis)

    # Compute edge features for rotated graph
    r_rotated = torch.sqrt(torch.sum(G.edata["d"]**2, -1, keepdim=True))
    feat_rotated = torch.cat([G.edata["w"], r_rotated],
                             -1) if "w" in G.edata else torch.cat([r_rotated],
                                                                  -1)

    # PairwiseConv forward pass for rotated graph
    output_rotated = pairwise_conv(feat_rotated, basis_rotated)

    # Test for equivariance under rotation
    output_diff = torch.norm(output_original -
                             output_rotated) / torch.norm(output_original)

    assert output_diff.item() < 1e-6


@pytest.mark.parametrize("batch_size, num_nodes, channels_0, channels_1", [
    (4, 10, 16, 32),
    (2, 5, 8, 16),
])
def test_se3_sum_layer(batch_size, num_nodes, channels_0, channels_1):
    """Test SE3Sum layer."""
    from deepchem.models.torch_models.layers import SE3Sum, Fiber
    f_x = Fiber(dictionary={0: channels_0, 1: channels_1})
    f_y = Fiber(dictionary={0: channels_0, 1: channels_1})
    gsum = SE3Sum(f_x, f_y)
    x = {
        '0': torch.randn(batch_size, num_nodes, channels_0, 1),
        '1': torch.randn(batch_size, num_nodes, channels_1, 3)
    }
    y = {
        '0': torch.randn(batch_size, num_nodes, channels_0, 1),
        '1': torch.randn(batch_size, num_nodes, channels_1, 3)
    }
    output = gsum(x, y)
    assert '0' in output and '1' in output
    assert output['0'].shape == x['0'].shape
    assert output['1'].shape == x['1'].shape
    assert torch.allclose(output['0'], x['0'] + y['0'])
    assert torch.allclose(output['1'], x['1'] + y['1'])


@pytest.mark.parametrize("batch_size, num_nodes, channels_0, channels_1", [
    (4, 10, 16, 32),
    (2, 5, 8, 16),
])
def test_se3_cat_layer(batch_size, num_nodes, channels_0, channels_1):
    """Test SE3Sum layer."""
    from deepchem.models.torch_models.layers import SE3Cat, Fiber
    f_x = Fiber(dictionary={0: channels_0, 1: channels_1})
    f_y = Fiber(dictionary={0: channels_0, 1: channels_1})
    gcat = SE3Cat(f_x, f_y)
    x = {
        '0': torch.randn(batch_size, num_nodes, channels_0, 1),
        '1': torch.randn(batch_size, num_nodes, channels_1, 3)
    }
    y = {
        '0': torch.randn(batch_size, num_nodes, channels_0, 1),
        '1': torch.randn(batch_size, num_nodes, channels_1, 3)
    }
    output = gcat(x, y)
    assert '0' in output and '1' in output
    assert output['0'].shape == (batch_size, 2 * num_nodes, channels_0, 1)
    assert output['1'].shape == (batch_size, 2 * num_nodes, channels_1, 3)


@pytest.mark.parametrize("batch_size, num_nodes, channels_0, channels_1", [
    (4, 10, 16, 32),
    (2, 6, 8, 16),
])
def test_se3_avgpooling_layer(batch_size, num_nodes, channels_0, channels_1):
    """Test SE3AvgPooling with scalar (degree 0) and vector (degree 1) features."""
    from deepchem.models.torch_models.layers import SE3AvgPooling
    import dgl

    # Create DGL graph
    G = dgl.graph(([0, 1, 2], [3, 4, 5]), num_nodes=num_nodes)

    # Random features
    features = {
        '0': torch.randn(num_nodes, channels_0),  # Scalars (Degree 0)
        '1':
            torch.randn(num_nodes, channels_1, 3)  # Vectors (Degree 1)
    }

    # Initialize Pooling Layers
    pool_0 = SE3AvgPooling(pooling_type='0')  # Scalars
    pool_1 = SE3AvgPooling(pooling_type='1')  # Vectors

    # Apply Pooling
    pooled_0 = pool_0(features, G)  # Scalar pooling
    pooled_1 = pool_1(features, G)  # Vector pooling

    # Expected output shapes
    expected_shape_0 = torch.randn(1).shape  # Adjust dynamically
    expected_shape_1 = torch.randn(1, channels_1, 3).shape  # Adjust dynamically

    assert pooled_0.shape == expected_shape_0

    assert pooled_1['1'].shape == expected_shape_1


@pytest.mark.torch
def test_equivariant_linear_module():
    """Test the EquivariantLinear layer transformations."""
    if not has_torch:
        pytest.skip("PyTorch is not installed.")

    from deepchem.models.torch_models.layers import EquivariantLinear

    B, N, D = 1, 16, 64
    layer = EquivariantLinear(D, D * 2).double()
    x = torch.randn(B, N, D, dtype=torch.float64)
    out = layer(x)
    assert out.shape == (B, N, D * 2)


@pytest.mark.torch
@pytest.mark.parametrize("batch_size, num_nodes, input_dim, output_dim", [
    (1, 16, 64, 128),
    (4, 32, 128, 64),
    (2, 1, 32, 16),
    (0, 10, 64, 128),
])
def test_equivariant_linear_module_initialization(batch_size, num_nodes,
                                                  input_dim, output_dim):
    """Test the EquivariantLinear layer initialization."""
    if not has_torch:
        pytest.skip("PyTorch is not installed.")

    from deepchem.models.torch_models.layers import EquivariantLinear

    layer = EquivariantLinear(input_dim, output_dim).double()
    x = torch.randn(batch_size, num_nodes, input_dim, dtype=torch.float64)
    out = layer(x)

    assert out.shape == (batch_size, num_nodes, output_dim)

    assert torch.isfinite(out).all()

    if batch_size > 0:
        out.sum().backward()
        assert layer.weight.grad is not None


@pytest.mark.torch
def test_se3_attention_module():
    """Test the SE(3) Attention mechanism."""
    if not has_torch:
        pytest.skip("PyTorch is not installed.")

    from deepchem.models.torch_models.layers import SE3Attention

    B, N, D = 1, 16, 64
    x = torch.randn(B, N, D, dtype=torch.float64)
    coords = torch.randn(B, N, 3, dtype=torch.float64)
    layer = SE3Attention(embed_dim=D, num_heads=4).double()
    out_x, out_coords = layer(x, coords)
    assert out_x.shape == x.shape
    assert out_coords.shape == coords.shape


@pytest.mark.torch
@pytest.mark.parametrize("batch_size, num_nodes, feature_dim", [
    (1, 16, 64),
    (4, 32, 128),
    (2, 1, 32),
])
def test_se3_attention_module_initialization(batch_size, num_nodes,
                                             feature_dim):
    """Test the SE(3) Attention mechanism initilization."""
    if not has_torch:
        pytest.skip("PyTorch is not installed.")

    from deepchem.models.torch_models.layers import SE3Attention

    x = torch.randn(batch_size, num_nodes, feature_dim, dtype=torch.float64)
    coords = torch.randn(batch_size, num_nodes, 3, dtype=torch.float64)
    layer = SE3Attention(embed_dim=feature_dim, num_heads=4).double()
    out_x, out_coords = layer(x, coords)

    assert out_x.shape == x.shape
    assert out_coords.shape == coords.shape

    assert torch.isfinite(out_x).all()
    assert torch.isfinite(out_coords).all()

    if batch_size > 0:
        out_x.sum().backward()
        assert layer.query.weight.grad is not None


@pytest.mark.torch
def test_se3_attention_equivariance():
    """Test SE(3) Attention for equivariance."""
    if not has_torch:
        pytest.skip("PyTorch is not installed.")

    from deepchem.models.torch_models.layers import SE3Attention

    B, N, D = 1, 16, 64
    x = torch.randn(B, N, D, dtype=torch.float64)
    coords = torch.randn(B, N, 3, dtype=torch.float64)
    layer = SE3Attention(embed_dim=D, num_heads=4).double()

    axis = np.array([0, 1, 0])
    angle = np.pi / 4
    rot_matrix = torch.tensor(rotation_matrix(axis, angle), dtype=torch.float64)
    rotated_coords = coords @ rot_matrix.T

    out_x_original, out_coords_original = layer(x, coords)
    out_x_rotated, out_coords_rotated = layer(x, rotated_coords)

    recovered_coords = out_coords_rotated @ rot_matrix
    assert torch.allclose(out_coords_original, recovered_coords, atol=1e-2)

    assert torch.allclose(out_x_original, out_x_rotated, atol=1e-2)


def test_fiber_initialization_from_degrees_channels():
    """Test Fiber initialization with num_degrees and num_channels."""
    from deepchem.models.torch_models.layers import Fiber
    fiber = Fiber(num_degrees=3, num_channels=16)
    expected_structure = [(16, 0), (16, 1), (16, 2)]
    assert fiber.structure == expected_structure
    assert fiber.n_features == np.sum(
        [i[0] * (2 * i[1] + 1) for i in expected_structure])


def test__fiber_initialization_from_dictionary():
    """Test Fiber initialization with a dictionary input."""
    from deepchem.models.torch_models.layers import Fiber
    fiber = Fiber(dictionary={0: 16, 1: 8, 2: 4})
    expected_structure = [(16, 0), (8, 1), (4, 2)]
    assert fiber.structure == expected_structure


def test_combine_fiber():
    """Test Fiber.combine() for correct summation of multiplicities."""
    from deepchem.models.torch_models.layers import Fiber
    fiber1 = Fiber(dictionary={0: 16, 1: 8})
    fiber2 = Fiber(dictionary={1: 8, 2: 4})
    combined = Fiber.combine(fiber1, fiber2)
    expected_structure = [(16, 0), (16, 1), (4, 2)]
    assert combined.structure == expected_structure


def test_combine_max_fiber():
    """Test Fiber.combine_max() for correct max operation on multiplicities."""
    from deepchem.models.torch_models.layers import Fiber
    fiber1 = Fiber(dictionary={0: 16, 1: 8})
    fiber2 = Fiber(dictionary={1: 12, 2: 4})
    combined_max = Fiber.combine_max(fiber1, fiber2)
    expected_structure = [(16, 0), (12, 1), (4, 2)]
    assert combined_max.structure == expected_structure


def test_feature_indices_fiber():
    """Test that feature indices are calculated correctly."""
    from deepchem.models.torch_models.layers import Fiber
    fiber = Fiber(dictionary={0: 2, 1: 2})
    expected_indices = {
        0: (0, 0 + 2 * (2 * 0 + 1)),
        1: (2, 2 + 2 * (2 * 1 + 1))
    }
    assert fiber.feature_indices == expected_indices
