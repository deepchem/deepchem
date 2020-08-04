"""
Featurizers for inorganic crystals.
"""

import numpy as np

from deepchem.feat import MaterialStructureFeaturizer, MaterialCompositionFeaturizer
from deepchem.utils import pad_array


class ElementPropertyFingerprint(MaterialCompositionFeaturizer):
  """
  Fingerprint of elemental properties from composition.

  Based on the data source chosen, returns properties and statistics
  (min, max, range, mean, standard deviation, mode) for a compound
  based on elemental stoichiometry. E.g., the average electronegativity
  of atoms in a crystal structure. The chemical fingerprint is a 
  vector of these statistics. For a full list of properties and statistics,
  see ``matminer.featurizers.composition.ElementProperty(data_source).feature_labels()``.

  This featurizer requires the optional dependencies pymatgen and
  matminer. It may be useful when only crystal compositions are available
  (and not 3D coordinates).

  See references [1]_ [2]_ [3]_ [4]_ for more details.

  References
  ----------
  .. [1] MagPie data: Ward, L. et al. npj Comput Mater 2, 16028 (2016).
	 https://doi.org/10.1038/npjcompumats.2016.28

  .. [2] Deml data: Deml, A. et al. Physical Review B 93, 085142 (2016).
	 10.1103/PhysRevB.93.085142

  .. [3] Matminer: Ward, L. et al. Comput. Mater. Sci. 152, 60-69 (2018).

  .. [4] Pymatgen: Ong, S.P. et al. Comput. Mater. Sci. 68, 314-319 (2013).

  Notes
  -----
  `NaN` feature values are automatically converted to 0 by this featurizer.  

  """

  def __init__(self, data_source='matminer'):
    """
    Parameters
    ----------
    data_source : {"matminer", "magpie", "deml"}
      Source for element property data.

    """

    self.data_source = data_source

  def _featurize(self, composition):
    """
    Calculate chemical fingerprint from crystal composition.

    Parameters
    ----------
    composition: pymatgen.Composition object
      Composition object.

    Returns
    -------
    feats: np.ndarray
      Vector of properties and statistics derived from chemical
      stoichiometry. Some values may be NaN.

    """
    try:
      from matminer.featurizers.composition import ElementProperty
    except ModuleNotFoundError:
      raise ValueError("This class requires matminer to be installed.")

    ep = ElementProperty.from_preset(self.data_source)

    try:
      feats = ep.featurize(composition)
    except:
      feats = []

    return np.nan_to_num(np.array(feats))


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

  """

  def __init__(self, max_atoms, flatten=True):
    """
    Parameters
    ----------
    max_atoms : int
      Maximum number of atoms for any crystal in the dataset. Used to
      pad the Coulomb matrix.
    flatten : bool (default True)
      Return flattened vector of matrix eigenvalues.

    """

    self.max_atoms = int(max_atoms)
    self.flatten = flatten

  def _featurize(self, struct):
    """
    Calculate sine Coulomb matrix from pymatgen structure.

    Parameters
    ----------
    struct : pymatgen.Structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.
      
    Returns
    -------
    features: np.ndarray
      2D sine Coulomb matrix with shape (max_atoms, max_atoms),
      or 1D matrix eigenvalues with shape (max_atoms,). 

    """

    try:
      from matminer.featurizers.structure import SineCoulombMatrix as SCM
    except ModuleNotFoundError:
      raise ValueError("This class requires matminer to be installed.")

    # Get full N x N SCM
    scm = SCM(flatten=False)
    sine_mat = scm.featurize(struct)

    if self.flatten:
      eigs, _ = np.linalg.eig(sine_mat)
      zeros = np.zeros((1, self.max_atoms))
      zeros[:len(eigs)] = eigs
      features = zeros
    else:
      features = pad_array(sine_mat, self.max_atoms)

    features = np.asarray(features)

    return features


class StructureGraphFeaturizer(MaterialStructureFeaturizer):
  """
  Calculate structure graph features for crystals.

  Based on the implementation in Crystal Graph Convolutional
  Neural Networks (CGCNN). The method constructs a crystal graph
  representation including atom features (atomic numbers) and bond
  features (neighbor distances). Neighbors are determined by searching
  in a sphere around atoms in the unit cell. A Gaussian filter is
  applied to neighbor distances. All units are in angstrom.  

  This featurizer requires the optional dependency pymatgen. It may
  be useful when 3D coordinates are available and when using graph 
  network models and crystal graph convolutional networks.

  See [1]_ for more details.

  References
  ----------
  .. [1] T. Xie and J. C. Grossman, Phys. Rev. Lett. 120, 2018.

  """

  def __init__(self, radius=8.0, max_neighbors=12, step=0.2):
    """
    Parameters
    ----------
    radius : float (default 8.0)
      Radius of sphere for finding neighbors of atoms in unit cell.
    max_neighbors : int (default 12)
      Maximum number of neighbors to consider when constructing graph.
    step : float (default 0.2)
      Step size for Gaussian filter.

    """

    self.radius = radius
    self.max_neighbors = int(max_neighbors)
    self.step = step

  def _featurize(self, struct):
    """
    Calculate crystal graph features from pymatgen structure.

    Parameters
    ----------
    struct : pymatgen.Structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    feats: np.array
      Atomic and bond features. Atomic features are atomic numbers 
      and bond features are Gaussian filtered interatomic distances.

    """

    features = self._get_structure_graph_features(struct)
    features = np.array(features)

    return features

  def _get_structure_graph_features(self, struct):
    """
    Calculate structure graph features from pymatgen structure.

    Parameters
    ----------
    struct : pymatgen.Structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    feats: tuple[np.array]
      atomic numbers, filtered interatomic distance tensor, and neighbor ids
    
    """

    atom_features = np.array([site.specie.Z for site in struct], dtype='int32')

    neighbors = struct.get_all_neighbors(self.radius, include_index=True)
    neighbors = [sorted(n, key=lambda x: x[1]) for n in neighbors]

    # Get list of lists of neighbor distances
    neighbor_features, neighbor_idx = [], []
    for neighbor in neighbors:
      if len(neighbor) < self.max_neighbors:
        neighbor_idx.append(
            list(map(lambda x: x[2], neighbor)) +
            [0] * (self.max_neighbors - len(neighbor)))
        neighbor_features.append(
            list(map(lambda x: x[1], neighbor)) +
            [self.radius + 1.] * (self.max_neighbors - len(neighbor)))
      else:
        neighbor_idx.append(
            list(map(lambda x: x[2], neighbor[:self.max_neighbors])))
        neighbor_features.append(
            list(map(lambda x: x[1], neighbor[:self.max_neighbors])))

    neighbor_features = np.array(neighbor_features)
    neighbor_idx = np.array(neighbor_idx)
    neighbor_features = self._gaussian_filter(neighbor_features)
    neighbor_features = np.vstack(neighbor_features)

    return (atom_features, neighbor_features, neighbor_idx)

  def _gaussian_filter(self, distances):
    """
    Apply Gaussian filter to an array of interatomic distances.

    Parameters
    ----------
    distances : np.array
      Matrix of distances of dimension (num atoms) x (max neighbors). 

    Returns
    -------
    expanded_distances: np.array 
      Expanded distance tensor after Gaussian filtering. Dimensionality
      is (num atoms) x (max neighbors) x (len(filt))
    
    """

    filt = np.arange(0, self.radius + self.step, self.step)

    # Increase dimension of distance tensor and apply filter
    expanded_distances = np.exp(
        -(distances[..., np.newaxis] - filt)**2 / self.step**2)

    return expanded_distances
