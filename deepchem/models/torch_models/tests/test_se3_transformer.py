import pytest
import numpy as np

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


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
