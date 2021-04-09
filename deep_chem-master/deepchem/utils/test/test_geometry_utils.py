import unittest
import numpy as np
from deepchem.utils import geometry_utils
from deepchem.utils.geometry_utils import unit_vector
from deepchem.utils.geometry_utils import angle_between
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.utils.geometry_utils import generate_random_unit_vector
from deepchem.utils.geometry_utils import generate_random_rotation_matrix
from deepchem.utils.geometry_utils import is_angle_within_cutoff


class TestGeometryUtils(unittest.TestCase):

  def test_generate_random_unit_vector(self):
    for _ in range(100):
      u = generate_random_unit_vector()
      # 3D vector with unit length
      self.assertEqual(u.shape, (3,))
      self.assertAlmostEqual(np.linalg.norm(u), 1.0)

  def test_generate_random_rotation_matrix(self):
    # very basic test, we check if rotations actually work in test_rotate_molecules
    for _ in range(100):
      m = generate_random_rotation_matrix()
      self.assertEqual(m.shape, (3, 3))

  def test_unit_vector(self):
    for _ in range(10):
      vector = np.random.rand(3)
      norm_vector = unit_vector(vector)
      self.assertAlmostEqual(np.linalg.norm(norm_vector), 1.0)

  def test_angle_between(self):
    for _ in range(10):
      v1 = np.random.rand(3,)
      v2 = np.random.rand(3,)
      angle = angle_between(v1, v2)
      self.assertLessEqual(angle, np.pi)
      self.assertGreaterEqual(angle, 0.0)
      self.assertAlmostEqual(angle_between(v1, v1), 0.0)
      self.assertAlmostEqual(angle_between(v1, -v1), np.pi)

  def test_is_angle_within_cutoff(self):
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    angle_cutoff = 10
    assert is_angle_within_cutoff(v1, v2, angle_cutoff)

  def test_compute_pairwise_distances(self):
    n1 = 10
    n2 = 50
    coords1 = np.random.rand(n1, 3)
    coords2 = np.random.rand(n2, 3)

    distance = compute_pairwise_distances(coords1, coords2)
    self.assertEqual(distance.shape, (n1, n2))
    self.assertTrue((distance >= 0).all())
    # random coords between 0 and 1, so the max possible distance in sqrt(2)
    self.assertTrue((distance <= 2.0**0.5).all())

    # check if correct distance metric was used
    coords1 = np.array([[0, 0, 0], [1, 0, 0]])
    coords2 = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    distance = compute_pairwise_distances(coords1, coords2)
    self.assertTrue((distance == [[1, 2, 3], [0, 1, 2]]).all())

  def test_compute_centroid(self):
    N = 10
    coords = np.random.rand(N, 3)
    centroid = geometry_utils.compute_centroid(coords)
    assert centroid.shape == (3,)

  def test_subract_centroid(self):
    N = 10
    coords = np.random.rand(N, 3)
    centroid = geometry_utils.compute_centroid(coords)
    new_coords = geometry_utils.subtract_centroid(coords, centroid)
    assert new_coords.shape == (N, 3)
    new_centroid = geometry_utils.compute_centroid(new_coords)
    assert new_centroid.shape == (3,)
    np.testing.assert_almost_equal(new_centroid, np.zeros_like(new_centroid))
