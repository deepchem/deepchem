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
def test_bn_layer():
    from deepchem.models.torch_models.layers import BN
    batch_size, num_channels = 10, 30
    layer = BN(num_channels)
    x = torch.randn(batch_size, num_channels)
    output = layer(x)

    assert output.shape == x.shape
    assert torch.allclose(output.mean(dim=-1),
                          torch.zeros(batch_size),
                          atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize("num_freq, in_dim, out_dim, edge_dim", [(5, 10, 15, 3),
                                                                 (2, 8, 16, 2)])
def test_radial_func(num_freq, in_dim, out_dim, edge_dim):
    from deepchem.models.torch_models.layers import RadialFunc
    layer = RadialFunc(num_freq, in_dim, out_dim, edge_dim)
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


def get_equivariant_basis(G, max_degree):
    """Compute SE(3) equivariant basis for molecular graph G."""
    from deepchem.utils.equivariance_utils import get_spherical_from_cartesian, precompute_sh, basis_transformation_Q_J
    distances = G.edata['d']
    r_ij = get_spherical_from_cartesian(distances)
    Y = precompute_sh(r_ij, 2 * max_degree)

    basis = {}

    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                Q_J = basis_transformation_Q_J(J, d_in, d_out).float().T
                K_J = torch.matmul(Y[J], Q_J)
                K_Js.append(K_J)
            size = (-1, 1, 2 * d_out + 1, 1, 2 * d_in + 1,
                    2 * min(d_in, d_out) + 1)
            basis[f"{d_in},{d_out}"] = torch.stack(K_Js, -1).view(*size)

    return basis


@pytest.mark.torch
@pytest.mark.parametrize("max_degree, nc_in, nc_out, edge_dim",
                         [(3, 32, 128, 5)])
def test_pairwiseconv_equivariance(max_degree, nc_in, nc_out, edge_dim):
    """Test SE(3) equivariance of PairwiseConv using a real molecular graph (CCO)."""
    from rdkit import Chem
    import dgl
    from deepchem.models.torch_models.layers import PairwiseConv

    # Load molecule and featurize
    mol = Chem.MolFromSmiles("CCO")
    featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True,
                                                    embeded=True)
    mol_graph = featurizer.featurize([mol])[0]

    # Initialize PairwiseConv layer
    pairwise_conv = PairwiseConv(degree_in=0,
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

    # Compute initial SE(3) equivariant basis
    basis = get_equivariant_basis(G, max_degree)

    # Compute radial distances and edge features before rotation
    r = torch.sqrt(torch.sum(G.edata["d"]**2, -1, keepdim=True))
    feat = torch.cat([G.edata["w"], r], -1) if "w" in G.edata else torch.cat(
        [r], -1)

    # Apply random graph rotation to nodes
    axis = np.array([1.0, 1.0, 1.0])  # Rotate around (1,1,1) axis
    angle = np.pi / 4  # 45-degree rotation
    G.ndata['x'] = apply_rotation(G.ndata['x'], axis, angle)  # Apply rotation

    # Compute SE(3) rotated equivariant basis
    basis_rotated = get_equivariant_basis(G, max_degree)

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
