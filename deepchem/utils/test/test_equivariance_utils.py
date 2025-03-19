import unittest
import math
import numpy as np
import deepchem as dc
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

if has_torch:
    from deepchem.utils import equivariance_utils


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
    from deepchem.utils.test.test_equivariance_utils import rotation_matrix
    R = torch.tensor(rotation_matrix(axis, angle), dtype=torch.float32)
    return torch.matmul(x, R.T)


class TestEquivarianceUtils(unittest.TestCase):
    """Test cases for the equivariance utilities."""

    def setUp(self):
        """Set up test fixtures."""
        import dgl
        from rdkit import Chem
        # Create a sample graph for testing
        # Load molecule and featurize
        mol = Chem.MolFromSmiles("CCO")
        featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True,
                                                        embeded=True)
        mol_graph = featurizer.featurize([mol])[0]

        self.G = dgl.graph((mol_graph.edge_index[0], mol_graph.edge_index[1]))
        self.G.ndata['f'] = torch.tensor(mol_graph.node_features,
                                         dtype=torch.float32).unsqueeze(-1)
        self.G.ndata['x'] = torch.tensor(
            mol_graph.positions, dtype=torch.float32)  # Atomic positions
        self.G.edata['d'] = torch.tensor(mol_graph.edge_features,
                                         dtype=torch.float32)
        self.G.edata['w'] = torch.tensor(mol_graph.edge_weights,
                                         dtype=torch.float32)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_generators_shape(self) -> None:
        # Test if the output has the correct shape.
        j_values = [1, 2, 3, 5,
                    7]  # Test for multiple quantum angular momentum values
        for j in j_values:
            with self.subTest(j=j):
                generators = equivariance_utils.su2_generators(j)
                self.assertEqual(generators.shape, (3, 2 * j + 1, 2 * j + 1))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_generators_zero_momenta(self) -> None:
        # Test for the case of zero momentum (j=0).
        j = 0
        generators = equivariance_utils.su2_generators(j)
        expected_generators = torch.zeros((3, 1, 1), dtype=torch.complex64)
        self.assertTrue(torch.allclose(generators, expected_generators))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_generators_angular_momentum_one(self) -> None:
        # Test for the case of momentum j=1 (spin-1).
        j = 1
        generators = equivariance_utils.su2_generators(j)
        # Expected J_x, J_z, J_y matrices for j=1

        expected_generators = torch.tensor(
            [[[0.0000 + 0.0000j, 0.7071 + 0.0000j, 0.0000 + 0.0000j],
              [-0.7071 + 0.0000j, 0.0000 + 0.0000j, 0.7071 + 0.0000j],
              [0.0000 + 0.0000j, -0.7071 + 0.0000j, 0.0000 + 0.0000j]],
             [[-0.0000 - 1.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
              [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
              [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 1.0000j]],
             [[0.0000 - 0.0000j, 0.0000 + 0.7071j, 0.0000 - 0.0000j],
              [0.0000 + 0.7071j, 0.0000 - 0.0000j, 0.0000 + 0.7071j],
              [0.0000 - 0.0000j, 0.0000 + 0.7071j, 0.0000 - 0.0000j]]])

        self.assertTrue(torch.allclose(generators, expected_generators))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_su2_commutation(self):
        j_values = [0, 0.5, 1, 1.5, 2,
                    2.5]  # Test for multiple quantum angular momentum values
        for j in j_values:
            with self.subTest(j=j):
                X = equivariance_utils.su2_generators(j)
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[0], X[1]),
                                   X[2]))
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[1], X[2]),
                                   X[0]))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_j_0(self):
        # Test for j = 0, which means we have a 1x1 transformation matrix
        j = 0
        Q = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex128)
        expected_Q = torch.tensor([[1]], dtype=torch.complex128)
        self.assertTrue(torch.allclose(Q, expected_Q))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_j_2(self):
        # Test for j = 2, which means we have a 5x5 transformation matrix
        j = 2
        Q = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex64)
        expected_Q = torch.tensor(
            [[
                0.0000 + 0.7071j, -0.0000 + 0.0000j, -0.0000 + 0.0000j,
                -0.0000 + 0.0000j, -0.7071 + 0.0000j
            ],
             [
                 -0.0000 + 0.0000j, 0.0000 + 0.7071j, -0.0000 + 0.0000j,
                 -0.7071 + 0.0000j, -0.0000 + 0.0000j
             ],
             [
                 -0.0000 + 0.0000j, -0.0000 + 0.0000j, -1.0000 + 0.0000j,
                 -0.0000 + 0.0000j, -0.0000 + 0.0000j
             ],
             [
                 -0.0000 + 0.0000j, 0.0000 + 0.7071j, -0.0000 + 0.0000j,
                 0.7071 - 0.0000j, -0.0000 + 0.0000j
             ],
             [
                 -0.0000 - 0.7071j, -0.0000 + 0.0000j, -0.0000 + 0.0000j,
                 -0.0000 + 0.0000j, -0.7071 + 0.0000j
             ]],
            dtype=torch.complex64)
        self.assertTrue(torch.allclose(Q, expected_Q))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_device(self) -> None:
        # Test for device placement (CPU to CUDA)
        j = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Q = equivariance_utils.change_basis_real_to_complex(j, device=device)
        self.assertEqual(Q.device, device)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_change_basis_real_to_complex_dtype_conversion(self):
        # Test for dtype conversion (complex128 to complex64)
        j = 1
        Q_complex128 = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex128)
        Q_complex64 = equivariance_utils.change_basis_real_to_complex(
            j, dtype=torch.complex64)

        self.assertEqual(Q_complex128.dtype, torch.complex128)
        self.assertEqual(Q_complex64.dtype, torch.complex64)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_generators_shape(self):
        j_values = [1, 2, 3, 4, 5]
        for j in j_values:
            with self.subTest(j=j):
                result = equivariance_utils.so3_generators(j)
                expected_shape = (3, 2 * j + 1, 2 * j + 1)
                self.assertEqual(result.shape, expected_shape)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_commutation(self):
        j_values = [0, 1, 2, 3, 4,
                    5]  # Test for multiple quantum angular momentum values
        for j in j_values:
            with self.subTest(j=j):
                X = equivariance_utils.so3_generators(j)
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[0], X[1]),
                                   X[2]))
                self.assertTrue(
                    torch.allclose(equivariance_utils.commutator(X[1], X[2]),
                                   X[0]))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_j0(self):
        j = 0
        result = equivariance_utils.so3_generators(j)
        expected = torch.tensor([[[0.]], [[0.]], [[0.]]],
                                dtype=torch.float64).float()
        self.assertTrue(torch.allclose(result, expected))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_so3_j1(self):
        j = 1
        result = equivariance_utils.so3_generators(j)
        expected = torch.tensor(
            [[[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, -1.0000],
              [0.0000, 1.0000, 0.0000]],
             [[0.0000, 0.0000, 1.0000], [0.0000, 0.0000, 0.0000],
              [-1.0000, 0.0000, 0.0000]],
             [[0.0000, -1.0000, 0.0000], [1.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]]],
            dtype=torch.float64).float()
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_unitary_property(self):
        j = 2
        alpha = torch.tensor([0.2])
        beta = torch.tensor([0.1])
        gamma = torch.tensor([0.7])

        D_matrix = equivariance_utils.wigner_D(j, alpha, beta, gamma)
        D_matrix = D_matrix[0]
        conjugate_transpose = torch.transpose(torch.conj(D_matrix), 0, 1)
        identity_matrix = torch.eye(D_matrix.shape[0], dtype=D_matrix.dtype)

        self.assertTrue(
            torch.allclose(D_matrix @ conjugate_transpose,
                           identity_matrix,
                           atol=1e-5))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_orthogonality(self):
        j = 2
        alpha = torch.tensor([0.2])
        beta = torch.tensor([0.1])
        gamma = torch.tensor([0.7])

        D_matrix = equivariance_utils.wigner_D(j, alpha, beta, gamma)
        D_matrix = D_matrix[0]
        num_columns = D_matrix.shape[1]

        for col1 in range(num_columns):
            for col2 in range(num_columns):
                if col1 != col2:
                    dot_product = torch.dot(D_matrix[:, col1], D_matrix[:,
                                                                        col2])
                    self.assertAlmostEqual(dot_product.item(), 0.0, places=5)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_semifactorial(self) -> None:
        # Test base cases
        self.assertEqual(equivariance_utils.semifactorial(0), 1.0)
        self.assertEqual(equivariance_utils.semifactorial(1), 1.0)

        self.assertEqual(equivariance_utils.semifactorial(5),
                         15.0)  # 5!! = 5 * 3 * 1
        self.assertEqual(equivariance_utils.semifactorial(6),
                         48.0)  # 6!! = 6 * 4 * 2

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_pochhammer_base_case(self) -> None:
        # Test k=0 (should return 1.0)
        self.assertEqual(equivariance_utils.pochhammer(3, 0), 1.0)

        self.assertEqual(equivariance_utils.pochhammer(3, 4),
                         360.0)  # (3)_4 = 3 * 4 * 5 * 6
        self.assertEqual(equivariance_utils.pochhammer(5, 2),
                         30.0)  # (5)_2 = 5 * 6

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_lpmv(self) -> None:
        x = torch.tensor([0.5])
        # Test P_2^1(x)
        result = equivariance_utils.lpmv(2, 1, x)
        expected = torch.tensor([-1.2990])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

        # Test P_1^1(x)
        result = equivariance_utils.lpmv(1, 1, x)
        expected = torch.tensor([-math.sqrt(1 - 0.5**2)])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

        # Test P_2^2(x)
        result = equivariance_utils.lpmv(2, 2, x)
        expected = torch.tensor([3 * (1 - 0.5**2)])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_harmonics(self) -> None:
        theta = torch.tensor([0.0, math.pi / 2])
        phi = torch.tensor([0.0, math.pi])
        sh = equivariance_utils.SphericalHarmonics()

        # Test Y_1^0(theta, phi)
        result = sh.get_element(1, 0, theta, phi)
        expected = torch.tensor([0.4886, -0.0000])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

        # Test Y_1^1(theta, phi)
        result = sh.get_element(1, 1, theta, phi)
        expected = torch.tensor([-0.0000, 0.4886])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

        # Get all spherical harmonics
        result = sh.get(1, theta, phi)
        expected = torch.tensor([[-0.0000, 0.4886, -0.0000],
                                 [0.0000, -0.0000, 0.4886]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_irr_repr(self) -> None:
        # Test irreducible representation of SO3.
        order = 1
        alpha = 0.1
        beta = 0.2
        gamma = 0.3

        # Edge case: order = 0.
        order_zero = 0
        result_zero = equivariance_utils.irr_repr(order_zero, alpha, beta,
                                                  gamma)
        self.assertTrue(torch.allclose(result_zero, torch.tensor([[1.0]])))

        result = equivariance_utils.irr_repr(order, alpha, beta, gamma)
        expected = torch.tensor([[0.9216, 0.0587, 0.3836],
                                 [0.0198, 0.9801, -0.1977],
                                 [-0.3875, 0.1898, 0.9021]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_get_matrix_kernel(self):
        # Test for computing the kernel of a matrix
        A = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
        kernel = equivariance_utils.get_matrix_kernel(A)
        for vector in kernel:
            result = torch.matmul(A, vector)
            self.assertTrue(
                torch.allclose(result, torch.zeros_like(result), atol=1e-5))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_basis_transformation_Q_J_output_shape(self):
        J, order_in, order_out = 1, 1, 1
        result = equivariance_utils.basis_transformation_Q_J(
            J, order_in, order_out)
        expected_shape = ((2 * order_out + 1) * (2 * order_in + 1), 2 * J + 1)
        self.assertEqual(result.shape, expected_shape)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_basis_transformation_Q_J_nonzero_output(self):
        J, order_in, order_out = 1, 1, 1
        result = equivariance_utils.basis_transformation_Q_J(
            J, order_in, order_out)
        self.assertTrue(torch.any(result != 0),
                        "Output tensor should not be all zeros")

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_kron(self):
        # Test Kronecker product
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[0.0, 5.0], [6.0, 7.0]])
        result = equivariance_utils.kron(A, B)
        expected = torch.tensor([[0.0, 5.0, 0.0, 10.0], [6.0, 7.0, 12.0, 14.0],
                                 [0.0, 15.0, 0.0, 20.0],
                                 [18.0, 21.0, 24.0, 28.0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_from_cartesian(self):
        cartesian = torch.tensor([[1.0, 1.0, 1.0]])  # [y, z, x]
        result = equivariance_utils.get_spherical_from_cartesian(cartesian)

        expected_radius = math.sqrt(1**2 + 1**2 + 1**2)
        expected_phi = math.pi / 4
        expected_theta = math.atan2(math.sqrt(1**2 + 1**2), 1)

        self.assertAlmostEqual(result[0, 0].item(), expected_radius, places=6)
        self.assertAlmostEqual(result[0, 1].item(), expected_phi, places=6)
        self.assertAlmostEqual(result[0, 2].item(), expected_theta, places=6)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_from_cartesian_zero_vector(self):
        cartesian = torch.tensor([[0.0, 0.0, 0.0]])
        result = equivariance_utils.get_spherical_from_cartesian(cartesian)

        self.assertAlmostEqual(result[0, 0].item(), 0.0,
                               places=6)  # Radius should be 0
        self.assertAlmostEqual(result[0, 1].item(), 0.0,
                               places=6)  # Azimuth phi should be 0
        self.assertAlmostEqual(result[0, 2].item(), 0.0,
                               places=6)  # Elevation theta should be 0

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_from_cartesian_negative(self):
        cartesian = torch.tensor([[-1.0, -1.0, -1.0]])
        result = equivariance_utils.get_spherical_from_cartesian(cartesian)
        expected_radius = math.sqrt(1**2 + 1**2 + 1**1)
        expected_phi = -3 * math.pi / 4
        expected_theta = math.atan2(math.sqrt(1**2 + 1**2), -1)

        self.assertAlmostEqual(result[0, 0].item(), expected_radius, places=6)
        self.assertAlmostEqual(result[0, 1].item(), expected_phi, places=6)
        self.assertAlmostEqual(result[0, 2].item(), expected_theta, places=6)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_from_cartesian_divide_radius_arg(self):
        cartesian = torch.tensor([[3.0, 4.0, 5.0]])
        result = equivariance_utils.get_spherical_from_cartesian(
            cartesian, divide_radius_by=2.0)
        expected_radius = math.sqrt(3**2 + 4**2 + 5**2) / 2

        self.assertAlmostEqual(result[0, 0].item(), expected_radius, places=6)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_from_cartesian_angle_boundaries(self):
        """Ensures φ is in [-π, π] and θ is in [0, π]"""
        cartesian = torch.tensor([[2.0, -2.0, 2.0]])
        result = equivariance_utils.get_spherical_from_cartesian(cartesian)

        phi = result[0, 1].item()
        theta = result[0, 2].item()

        self.assertTrue(-math.pi <= phi <= math.pi,
                        msg=f"Azimuth φ out of bounds: {phi}")
        self.assertTrue(0 <= theta <= math.pi,
                        msg=f"Elevation θ out of bounds: {theta}")

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_spherical_from_cartesian_high_dimensional_tensor(self):
        cartesian = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                                  [[0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]])
        result = equivariance_utils.get_spherical_from_cartesian(cartesian)

        self.assertEqual(result.shape, (2, 2, 3))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_precompute_sh(self):
        """Test spherical harmonics computation for a simple input."""
        r_ij = torch.tensor([[1.0, 0.5, 1.0]])  # [radius, phi, theta]
        max_J = 2
        result = equivariance_utils.precompute_sh(r_ij, max_J)

        self.assertEqual(len(result), max_J + 1)
        for J in range(max_J + 1):
            self.assertTrue(isinstance(result[J], torch.Tensor))

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_precompute_sh_zero_order(self):
        """Test when max_J = 0, only J=0 should be present."""
        r_ij = torch.tensor([[1.0, 1.0, 1.0]])
        result = equivariance_utils.precompute_sh(r_ij, max_J=0)

        self.assertEqual(len(result), 1)  # Only J=0 should be present
        self.assertIn(0, result)  # J=0 key should exist

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_precompute_sh_multiple_inputs(self):
        """Test batch inputs to check if function handles multiple r_ij vectors."""
        r_ij = torch.tensor([[1.0, 0.5, 1.0], [2.0, 1.0, 0.5]])
        max_J = 3
        result = equivariance_utils.precompute_sh(r_ij, max_J)

        self.assertEqual(len(result), max_J + 1)
        for J in range(max_J + 1):
            self.assertEqual(result[J].shape[0], r_ij.shape[0])

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_precompute_sh_output_shape(self):
        """Test if each J value has shape [B, N, K, 2J+1]."""
        r_ij = torch.tensor([[1.0, 0.5, 1.0], [2.0, 1.0, 0.5]])
        max_J = 2
        result = equivariance_utils.precompute_sh(r_ij, max_J)

        for J in range(max_J + 1):
            expected_last_dim = 2 * J + 1
            self.assertEqual(result[J].shape[-1], expected_last_dim)

    def sample_graph(self):
        """Create a test graph with SE(3) features using the SMILES 'CCO'."""
        import dgl
        from rdkit import Chem
        import deepchem as dc

        mol = Chem.MolFromSmiles("CCO")
        featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True,
                                                        embeded=True)
        features = featurizer.featurize([mol])[0]

        G = dgl.graph((features.edge_index[0], features.edge_index[1]),
                      num_nodes=len(features.node_features))
        G.ndata['x'] = torch.tensor(features.positions, dtype=torch.float32)
        G.edata['d'] = G.ndata['x'][G.edges()[1]] - G.ndata['x'][G.edges()[0]]

        return G

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_translation_equivariance(self):
        """Check that translating node positions does not change basis functions or relative distances."""
        from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r

        basis_original, r_original = get_equivariant_basis_and_r(
            self.G, max_degree=2, compute_gradients=False)

        # Apply random translation
        translation = torch.randn(1, 3)
        G_translated = self.G.clone()
        G_translated.ndata['x'] += translation
        G_translated.edata['d'] = G_translated.ndata['x'][G_translated.edges(
        )[1]] - G_translated.ndata['x'][G_translated.edges()[0]]

        basis_translated, r_translated = get_equivariant_basis_and_r(
            G_translated, max_degree=2, compute_gradients=False)

        for key in basis_original:
            if key == '0,0':
                assert torch.allclose(
                    basis_original[key], basis_translated[key],
                    atol=1e-5), f"Failed translation equivariance for {key}"

        # r should be unchanged under translation
        assert torch.allclose(r_original, r_translated, atol=1e-5)

    @unittest.skipIf(not has_torch, "torch is not available")
    def test_rotation_equivariance(self):
        """Check that rotating node positions results in equivalent basis transformations and correct `r` behavior."""
        from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r

        basis_original, r_original = get_equivariant_basis_and_r(
            self.G, max_degree=2, compute_gradients=False)

        # Apply rotation
        axis = torch.randn(3)
        angle = torch.rand(1).item() * 2 * np.pi

        G_rotated = self.G.clone()
        G_rotated.ndata['x'] = apply_rotation(self.G.ndata['x'], axis, angle)
        G_rotated.edata['d'] = G_rotated.ndata['x'][
            G_rotated.edges()[1]] - G_rotated.ndata['x'][G_rotated.edges()[0]]

        basis_rotated, r_rotated = get_equivariant_basis_and_r(
            G_rotated, max_degree=2, compute_gradients=False)

        for key in basis_original:
            if key == '0,0':
                assert torch.allclose(
                    basis_original[key], basis_rotated[key],
                    atol=1e-5), f"Failed rotation equivariance for {key}"

        # Check if the norm of `r` is unchanged under rotation
        assert torch.allclose(r_original[..., 0], r_rotated[..., 0], atol=1e-5)
