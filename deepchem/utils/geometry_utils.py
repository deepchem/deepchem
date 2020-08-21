"""
Geometric utility functions for 3D geometry.
"""
import numpy as np
from scipy.spatial.distance import cdist


def unit_vector(vector: np.ndarray) -> np.ndarray:
  """ Returns the unit vector of the vector.

  Parameters
  ----------
  vector: np.ndarray
    A numpy array of shape `(3,)`, where `3` is (x,y,z).

  Returns
  ----------
  np.ndarray
    A numpy array of shape `(3,)`. The unit vector of the input vector.
  """
  return vector / np.linalg.norm(vector)


def angle_between(vector_i: np.ndarray, vector_j: np.ndarray) -> np.ndarray:
  """Returns the angle in radians between vectors "vector_i" and "vector_j"

  Note that this function always returns the smaller of the two angles between
  the vectors (value between 0 and pi).

  Parameters
  ----------
  vector_i: np.ndarray
    A numpy array of shape `(3,)`, where `3` is (x,y,z).
  vector_j: np.ndarray
    A numpy array of shape `(3,)`, where `3` is (x,y,z).

  Returns
  ----------
  np.ndarray
    The angle in radians between the two vectors.

  Examples
  --------
  >>> print("%0.06f" % angle_between((1, 0, 0), (0, 1, 0)))
  1.570796
  >>> print("%0.06f" % angle_between((1, 0, 0), (1, 0, 0)))
  0.000000
  >>> print("%0.06f" % angle_between((1, 0, 0), (-1, 0, 0)))
  3.141593
  """
  vector_i_u = unit_vector(vector_i)
  vector_j_u = unit_vector(vector_j)
  angle = np.arccos(np.dot(vector_i_u, vector_j_u))
  if np.isnan(angle):
    if np.allclose(vector_i_u, vector_j_u):
      return 0.0
    else:
      return np.pi
  return angle


def generate_random_unit_vector() -> np.ndarray:
  """Generate a random unit vector on the sphere S^2.

  Citation: http://mathworld.wolfram.com/SpherePointPicking.html

  Pseudocode:
    a. Choose random theta \element [0, 2*pi]
    b. Choose random z \element [-1, 1]
    c. Compute output vector u: (x,y,z) = (sqrt(1-z^2)*cos(theta), sqrt(1-z^2)*sin(theta),z)

  Returns
  -------
  u: np.ndarray
    A numpy array of shape `(3,)`. u is an unit vector
  """
  theta = np.random.uniform(low=0.0, high=2 * np.pi)
  z = np.random.uniform(low=-1.0, high=1.0)
  u = np.array(
      [np.sqrt(1 - z**2) * np.cos(theta),
       np.sqrt(1 - z**2) * np.sin(theta), z])
  return u


def generate_random_rotation_matrix() -> np.ndarray:
  """Generates a random rotation matrix.

  1. Generate a random unit vector u, randomly sampled from the
     unit sphere (see function generate_random_unit_vector()
     for details)
  2. Generate a second random unit vector v
    a. If absolute value of u \dot v > 0.99, repeat.
       (This is important for numerical stability. Intuition: we
       want them to be as linearly independent as possible or
       else the orthogonalized version of v will be much shorter
       in magnitude compared to u. I assume in Stack they took
       this from Gram-Schmidt orthogonalization?)
    b. v" = v - (u \dot v)*u, i.e. subtract out the component of
       v that's in u's direction
    c. normalize v" (this isn"t in Stack but I assume it must be
       done)
  3. find w = u \cross v"
  4. u, v", and w will form the columns of a rotation matrix, R.
     The intuition is that u, v" and w are, respectively, what
     the standard basis vectors e1, e2, and e3 will be mapped
     to under the transformation.

  Returns
  -------
  R: np.ndarray
    A numpy array of shape `(3, 3)`. R is a rotation matrix.
  """
  u = generate_random_unit_vector()
  v = generate_random_unit_vector()
  while np.abs(np.dot(u, v)) >= 0.99:
    v = generate_random_unit_vector()

  vp = v - (np.dot(u, v) * u)
  vp /= np.linalg.norm(vp)
  w = np.cross(u, vp)
  R = np.column_stack((u, vp, w))
  return R


def is_angle_within_cutoff(vector_i: np.ndarray, vector_j: np.ndarray,
                           angle_cutoff: float) -> bool:
  """A utility function to compute whether two vectors are within a cutoff from 180 degrees apart.

  Parameters
  ----------
  vector_i: np.ndarray
    A numpy array of shape (3,)`, where `3` is (x,y,z).
  vector_j: np.ndarray
    A numpy array of shape `(3,)`, where `3` is (x,y,z).
  cutoff: float
    The deviation from 180 (in degrees)

  Returns
  -------
  bool
    Whether two vectors are within a cutoff from 180 degrees apart
  """
  angle = angle_between(vector_i, vector_j) * 180. / np.pi
  return (angle > (180 - angle_cutoff) and angle < (180. + angle_cutoff))


def compute_centroid(coordinates: np.ndarray) -> np.ndarray:
  """Compute the (x,y,z) centroid of provided coordinates

  Parameters
  ----------
  coordinates: np.ndarray
    A numpy array of shape `(N, 3)`, where `N` is the number of atoms.

  Returns
  -------
  centroid: np.ndarray
    A numpy array of shape `(3,)`, where `3` is (x,y,z).
  """
  centroid = np.mean(coordinates, axis=0)
  return centroid


def compute_protein_range(coordinates: np.ndarray) -> np.ndarray:
  """Compute the protein range of provided coordinates

  Parameters
  ----------
  coordinates: np.ndarray
    A numpy array of shape `(N, 3)`, where `N` is the number of atoms.

  Returns
  -------
  protein_range: np.ndarray
    A numpy array of shape `(3,)`, where `3` is (x,y,z).
  """
  protein_max = np.max(coordinates, axis=0)
  protein_min = np.min(coordinates, axis=0)
  protein_range = protein_max - protein_min
  return protein_range


def subtract_centroid(coordinates: np.ndarray,
                      centroid: np.ndarray) -> np.ndarray:
  """Subtracts centroid from each coordinate.

  Subtracts the centroid, a numpy array of dim 3, from all coordinates
  of all atoms in the molecule

  Note that this update is made in place to the array it's applied to.

  Parameters
  ----------
  coordinates: np.ndarray
    A numpy array of shape `(N, 3)`, where `N` is the number of atoms.
  centroid: np.ndarray
    A numpy array of shape `(3,)`

  Returns
  -------
  coordinates: np.ndarray
    A numpy array of shape `(3,)`, where `3` is (x,y,z).
  """
  coordinates -= np.transpose(centroid)
  return coordinates


def compute_pairwise_distances(first_coordinate: np.ndarray,
                               second_coordinate: np.ndarray) -> np.ndarray:
  """Computes pairwise distances between two molecules.

  Takes an input (m, 3) and (n, 3) numpy arrays of 3D coords of
  two molecules respectively, and outputs an m x n numpy
  array of pairwise distances in Angstroms between the first and
  second molecule. entry (i,j) is dist between the i"th
  atom of first molecule and the j"th atom of second molecule.

  Parameters
  ----------
  first_coordinate: np.ndarray
    A numpy array of shape `(m, 3)`, where `m` is the number of atoms.
  second_coordinate: np.ndarray
    A numpy array of shape `(n, 3)`, where `n` is the number of atoms.

  Returns
  -------
  pairwise_distances: np.ndarray
    A numpy array of shape `(m, n)`
  """

  pairwise_distances = cdist(
      first_coordinate, second_coordinate, metric='euclidean')
  return pairwise_distances
