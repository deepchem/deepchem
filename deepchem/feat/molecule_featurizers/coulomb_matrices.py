"""
Generate coulomb matrices for molecules.

See Montavon et al., _New Journal of Physics_ __15__ (2013) 095003.
"""
import numpy as np
from typing import Any, List, Optional

from deepchem.utils.typing import RDKitMol
from deepchem.utils.data_utils import pad_array
from deepchem.feat.base_classes import MolecularFeaturizer


class CoulombMatrix(MolecularFeaturizer):
  """Calculate Coulomb matrices for molecules.

  Coulomb matrices provide a representation of the electronic structure of a
  molecule. This method is described in [1]_.

  Examples
  --------
  >>> import deepchem as dc
  >>> featurizers = dc.feat.CoulombMatrix(max_atoms=23)
  >>> input_file = 'deepchem/feat/tests/data/water.sdf' # really backed by water.sdf.csv
  >>> tasks = ["atomization_energy"]
  >>> loader = dc.data.SDFLoader(tasks, featurizer=featurizers)
  >>> dataset = loader.create_dataset(input_file)


  References
  ----------
  .. [1] Montavon, Grégoire, et al. "Learning invariant representations of
     molecules for atomization energy prediction." Advances in neural information
     processing systems. 2012.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self,
               max_atoms: int,
               remove_hydrogens: bool = False,
               randomize: bool = False,
               upper_tri: bool = False,
               n_samples: int = 1,
               seed: Optional[int] = None):
    """Initialize this featurizer.

    Parameters
    ----------
    max_atoms: int
      The maximum number of atoms expected for molecules this featurizer will
      process.
    remove_hydrogens: bool, optional (default False)
      If True, remove hydrogens before processing them.
    randomize: bool, optional (default False)
      If True, use method `randomize_coulomb_matrices` to randomize Coulomb matrices.
    upper_tri: bool, optional (default False)
      Generate only upper triangle part of Coulomb matrices.
    n_samples: int, optional (default 1)
      If `randomize` is set to True, the number of random samples to draw.
    seed: int, optional (default None)
      Random seed to use.
    """
    self.max_atoms = int(max_atoms)
    self.remove_hydrogens = remove_hydrogens
    self.randomize = randomize
    self.upper_tri = upper_tri
    self.n_samples = n_samples
    if seed is not None:
      seed = int(seed)
    self.seed = seed

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """
    Calculate Coulomb matrices for molecules. If extra randomized
    matrices are generated, they are treated as if they are features
    for additional conformers.

    Since Coulomb matrices are symmetric, only the (flattened) upper
    triangular portion is returned.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      The coulomb matrices of the given molecule.
      The default shape is `(num_confs, max_atoms, max_atoms)`.
      If num_confs == 1, the shape is `(max_atoms, max_atoms)`.
    """
    features = self.coulomb_matrix(mol)
    if self.upper_tri:
      features = [f[np.triu_indices_from(f)] for f in features]
    features = np.asarray(features)
    if features.shape[0] == 1:
      # `(1, max_atoms, max_atoms)` -> `(max_atoms, max_atoms)`
      features = np.squeeze(features, axis=0)
    return features

  def coulomb_matrix(self, mol: RDKitMol) -> np.ndarray:
    """
    Generate Coulomb matrices for each conformer of the given molecule.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      The coulomb matrices of the given molecule
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import AllChem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    # Check whether num_confs >=1 or not
    num_confs = len(mol.GetConformers())
    if num_confs == 0:
      mol = Chem.AddHs(mol)
      AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    if self.remove_hydrogens:
      mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    rval = []
    for conf in mol.GetConformers():
      d = self.get_interatomic_distances(conf)
      m = np.outer(z, z) / d
      m[range(n_atoms), range(n_atoms)] = 0.5 * np.array(z)**2.4
      if self.randomize:
        for random_m in self.randomize_coulomb_matrix(m):
          random_m = pad_array(random_m, self.max_atoms)
          rval.append(random_m)
      else:
        m = pad_array(m, self.max_atoms)
        rval.append(m)
    return np.asarray(rval)

  def randomize_coulomb_matrix(self, m: np.ndarray) -> List[np.ndarray]:
    """Randomize a Coulomb matrix as decribed in [1]_:

    1. Compute row norms for M in a vector row_norms.
    2. Sample a zero-mean unit-variance noise vector e with dimension
       equal to row_norms.
    3. Permute the rows and columns of M with the permutation that
       sorts row_norms + e.

    Parameters
    ----------
    m: np.ndarray
      Coulomb matrix.

    Returns
    -------
    List[np.ndarray]
      List of the random coulomb matrix

    References
    ----------
    .. [1] Montavon et al., New Journal of Physics, 15, (2013), 095003
    """
    rval = []
    row_norms = np.asarray([np.linalg.norm(row) for row in m], dtype=float)
    rng = np.random.RandomState(self.seed)
    for i in range(self.n_samples):
      e = rng.normal(size=row_norms.size)
      p = np.argsort(row_norms + e)
      new = m[p][:, p]  # permute rows first, then columns
      rval.append(new)
    return rval

  @staticmethod
  def get_interatomic_distances(conf: Any) -> np.ndarray:
    """
    Get interatomic distances for atoms in a molecular conformer.

    Parameters
    ----------
    conf: rdkit.Chem.rdchem.Conformer
      Molecule conformer.

    Returns
    -------
    np.ndarray
      The distances matrix for all atoms in a molecule
    """
    n_atoms = conf.GetNumAtoms()
    coords = [
        # Convert AtomPositions from Angstrom to bohr (atomic units)
        conf.GetAtomPosition(i).__idiv__(0.52917721092) for i in range(n_atoms)
    ]
    d = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
      for j in range(i):
        d[i, j] = coords[i].Distance(coords[j])
        d[j, i] = d[i, j]
    return d


class CoulombMatrixEig(CoulombMatrix):
  """Calculate the eigenvalues of Coulomb matrices for molecules.

  This featurizer computes the eigenvalues of the Coulomb matrices for provided
  molecules. Coulomb matrices are described in [1]_.

  Examples
  --------
  >>> import deepchem as dc
  >>> featurizers = dc.feat.CoulombMatrixEig(max_atoms=23)
  >>> input_file = 'deepchem/feat/tests/data/water.sdf' # really backed by water.sdf.csv
  >>> tasks = ["atomization_energy"]
  >>> loader = dc.data.SDFLoader(tasks, featurizer=featurizers)
  >>> dataset = loader.create_dataset(input_file)

  References
  ----------
  .. [1] Montavon, Grégoire, et al. "Learning invariant representations of
     molecules for atomization energy prediction." Advances in neural information
     processing systems. 2012.
  """

  def __init__(self,
               max_atoms: int,
               remove_hydrogens: bool = False,
               randomize: bool = False,
               n_samples: int = 1,
               seed: Optional[int] = None):
    """Initialize this featurizer.

    Parameters
    ----------
    max_atoms: int
      The maximum number of atoms expected for molecules this featurizer will
      process.
    remove_hydrogens: bool, optional (default False)
      If True, remove hydrogens before processing them.
    randomize: bool, optional (default False)
      If True, use method `randomize_coulomb_matrices` to randomize Coulomb matrices.
    n_samples: int, optional (default 1)
      If `randomize` is set to True, the number of random samples to draw.
    seed: int, optional (default None)
      Random seed to use.
    """
    self.max_atoms = int(max_atoms)
    self.remove_hydrogens = remove_hydrogens
    self.randomize = randomize
    self.n_samples = n_samples
    if seed is not None:
      seed = int(seed)
    self.seed = seed

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """
    Calculate eigenvalues of Coulomb matrix for molecules. Eigenvalues
    are returned sorted by absolute value in descending order and padded
    by max_atoms.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      The eigenvalues of Coulomb matrix for molecules.
      The default shape is `(num_confs, max_atoms)`.
      If num_confs == 1, the shape is `(max_atoms,)`.
    """
    cmat = self.coulomb_matrix(mol)
    features_list = []
    for f in cmat:
      w, v = np.linalg.eig(f)
      w_abs = np.abs(w)
      sortidx = np.argsort(w_abs)
      sortidx = sortidx[::-1]
      w = w[sortidx]
      f = pad_array(w, self.max_atoms)
      features_list.append(f)
    features = np.asarray(features_list)
    if features.shape[0] == 1:
      # `(1, max_atoms)` -> `(max_atoms,)`
      features = np.squeeze(features, axis=0)
    return features
