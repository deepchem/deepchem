"""
Utilities to score protein-ligand poses using DeepChem.
"""
import numpy as np


def pairwise_distances(coords1, coords2):
  """Returns matrix of pairwise Euclidean distances.

  Parameters
  ----------
  coords1: np.ndarray
    Of shape `(N, 3)`
  coords2: np.ndarray
    Of shape `(M, 3)`

  Returns
  -------
  A `(N,M)` array with pairwise distances.
  """
  return np.sum((coords1[None, :] - coords2[:, None])**2, -1)**0.5


def cutoff_filter(d, x, cutoff=8.0):
  """Applies a cutoff filter on pairwise distances

  Parameters
  ----------
  d: np.ndarray
    Pairwise distances matrix. Of shape `(N, M)` 
  x: np.ndarray
    Matrix of shape `(N, M)` 
  cutoff: float, optional (default 8)
    Cutoff for selection in Angstroms

  Returns
  -------
  A `(N,M)` array with values where distance is too large thresholded
  to 0.
  """
  return np.where(d < cutoff, x, np.zeros_like(x))


def vina_nonlinearity(c, w, Nrot):
  """Computes non-linearity used in Vina.

  Parameters
  ----------
  c: np.ndarray 
    Of shape `(N, M)` 
  w: float
    Weighting term
  Nrot: int
    Number of rotatable bonds in this molecule

  Returns
  -------
  A `(N, M)` array with activations under a nonlinearity.
  """
  out_tensor = c / (1 + w * Nrot)
  return out_tensor


def vina_repulsion(d):
  """Computes Autodock Vina's repulsion interaction term.

  Parameters
  ----------
  d: np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array with repulsion terms.
  """
  return np.where(d < 0, d**2, np.zeros_like(d))


def vina_hydrophobic(d):
  """Computes Autodock Vina's hydrophobic interaction term.

  Here, d is the set of surface distances as defined in:

  Jain, Ajay N. "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of computer-aided molecular design 10.5 (1996): 427-440.

  Parameters
  ----------
  d: np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of hydrophoboic interactions in a piecewise linear
  curve.
  """
  out_tensor = np.where(d < 0.5, np.ones_like(d),
                        np.where(d < 1.5, 1.5 - d, np.zeros_like(d)))
  return out_tensor


def vina_hbond(d):
  """Computes Autodock Vina's hydrogen bond interaction term.

  Here, d is the set of surface distances as defined in:

  Jain, Ajay N. "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of computer-aided molecular design 10.5 (1996): 427-440.

  Parameters
  ----------
  d: np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of hydrophoboic interactions in a piecewise linear
  curve.
  """
  out_tensor = np.where(
      d < -0.7, np.ones_like(d),
      np.where(d < 0, (1.0 / 0.7) * (0 - d), np.zeros_like(d)))
  return out_tensor


def vina_gaussian_first(d):
  """Computes Autodock Vina's first Gaussian interaction term.

  Here, d is the set of surface distances as defined in:

  Jain, Ajay N. "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of computer-aided molecular design 10.5 (1996): 427-440.

  Parameters
  ----------
  d: np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of gaussian interaction terms.
  """
  out_tensor = np.exp(-(d / 0.5)**2)
  return out_tensor


def vina_gaussian_second(d):
  """Computes Autodock Vina's second Gaussian interaction term.

  Here, d is the set of surface distances as defined in:

  Jain, Ajay N. "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of computer-aided molecular design 10.5 (1996): 427-440.

  Parameters
  ----------
  d: np.ndarray
    Of shape `(N, M)`.

  Returns
  -------
  A `(N, M)` array of gaussian interaction terms.
  """
  out_tensor = np.exp(-((d - 3) / 2)**2)
  return out_tensor


def weighted_linear_sum(w, x):
  """Computes weighted linear sum.

  Parameters
  ----------
  w: np.ndarray
    Of shape `(N,)`
  x: np.ndarray
    Of shape `(N,)`
  """
  return np.sum(np.dot(w, x))


def vina_energy_term(coords1, coords2, weights, wrot, Nrot):
  """Computes the Vina Energy function for two molecular conformations

  Parameters
  ----------
  coords1: np.ndarray 
    Molecular coordinates of shape `(N, 3)`
  coords2: np.ndarray 
    Molecular coordinates of shape `(M, 3)`
  weights: np.ndarray
    Of shape `(5,)`
  wrot: float
    The scaling factor for nonlinearity
  Nrot: int
    Number of rotatable bonds in this calculation

  Returns
  -------
  Scalar with energy
  """
  # TODO(rbharath): The autodock vina source computes surface distances which take into account the van der Waals radius of each atom type.
  dists = pairwise_distances(coords1, coords2)
  repulsion = vina_repulsion(dists)
  hydrophobic = vina_hydrophobic(dists)
  hbond = vina_hbond(dists)
  gauss_1 = vina_gaussian_first(dists)
  gauss_2 = vina_gaussian_second(dists)

  # Shape (N, M)
  interactions = weighted_linear_sum(
      weights, np.array([repulsion, hydrophobic, hbond, gauss_1, gauss_2]))

  # Shape (N, M)
  thresholded = cutoff_filter(dists, interactions)

  free_energies = vina_nonlinearity(thresholded, wrot, Nrot)
  return np.sum(free_energies)
