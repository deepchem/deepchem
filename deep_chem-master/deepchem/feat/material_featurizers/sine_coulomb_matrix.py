import numpy as np

from deepchem.utils.typing import PymatgenStructure
from deepchem.feat import MaterialStructureFeaturizer
from deepchem.utils.data_utils import pad_array
from typing import Any


class SineCoulombMatrix(MaterialStructureFeaturizer):
  """
  Calculate sine Coulomb matrix for crystals.

  A variant of Coulomb matrix for periodic crystals.

  The sine Coulomb matrix is identical to the Coulomb matrix, except
  that the inverse distance function is replaced by the inverse of
  sin**2 of the vector between sites which are periodic in the
  dimensions of the crystal lattice.

  Features are flattened into a vector of matrix eigenvalues by default
  for ML-readiness. To ensure that all feature vectors are equal
  length, the maximum number of atoms (eigenvalues) in the input
  dataset must be specified.

  This featurizer requires the optional dependencies pymatgen and
  matminer. It may be useful when crystal structures with 3D coordinates
  are available.

  See [1]_ for more details.

  References
  ----------
  .. [1] Faber et al. Inter. J. Quantum Chem. 115, 16, 2015.

  Examples
  --------
  >>> import pymatgen as mg
  >>> lattice = mg.core.Lattice.cubic(4.2)
  >>> structure = mg.core.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
  >>> featurizer = SineCoulombMatrix(max_atoms=2)
  >>> features = featurizer.featurize([structure])

  Note
  ----
  This class requires matminer and Pymatgen to be installed.
  """

  def __init__(self, max_atoms: int = 100, flatten: bool = True):
    """
    Parameters
    ----------
    max_atoms: int (default 100)
      Maximum number of atoms for any crystal in the dataset. Used to
      pad the Coulomb matrix.
    flatten: bool (default True)
      Return flattened vector of matrix eigenvalues.
    """
    self.max_atoms = max_atoms
    self.flatten = flatten
    self.scm: Any = None

  def _featurize(self, struct: PymatgenStructure) -> np.ndarray:
    """
    Calculate sine Coulomb matrix from pymatgen structure.

    Parameters
    ----------
    struct: pymatgen.core.Structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    features: np.ndarray
      2D sine Coulomb matrix with shape (max_atoms, max_atoms),
      or 1D matrix eigenvalues with shape (max_atoms,).
    """
    if self.scm is None:
      try:
        from matminer.featurizers.structure import SineCoulombMatrix as SCM
        self.scm = SCM(flatten=False)
      except ModuleNotFoundError:
        raise ImportError("This class requires matminer to be installed.")

    # Get full N x N SCM
    sine_mat = self.scm.featurize(struct)

    if self.flatten:
      eigs, _ = np.linalg.eig(sine_mat)
      zeros = np.zeros(self.max_atoms)
      zeros[:len(eigs[0])] = eigs[0]
      features = zeros
    else:
      features = pad_array(sine_mat, self.max_atoms)

    features = np.asarray(features)

    return features
