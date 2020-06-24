"""
Featurizers for inorganic crystals.
"""

import numpy as np

from deepchem.feat import Featurizer
from deepchem.utils import pad_array


class ChemicalFingerprint(Featurizer):
  """
  Chemical fingerprint of elemental properties from composition.

  Based on the data source chosen, returns properties and statistics
  (min, max, range, mean, standard deviation, mode) for a compound
  based on elemental stoichiometry. E.g., the average electronegativity
  of atoms in a crystal structure. The chemical fingerprint is a 
  vector of these statistics. For a full list of properties and statistics,
  see ``matminer.featurizers.composition.ElementProperty(data_source).feature_labels()``.

  This featurizer requires the optional dependencies pymatgen and
  matminer. It may be useful when only crystal compositions are available
  (and not 3D coordinates).

  References are given for each data source:
    MagPie data: Ward, L. et al. npj Comput Mater 2, 16028 (2016).
    https://doi.org/10.1038/npjcompumats.2016.28

    Deml data: Deml, A. et al. Physical Review B 93, 085142 (2016).
    10.1103/PhysRevB.93.085142

    Matminer: Ward, L. et al. Comput. Mater. Sci. 152, 60-69 (2018).

    Pymatgen: Ong, S.P. et al. Comput. Mater. Sci. 68, 314-319 (2013). 

  """

  def __init__(self, data_source='matminer'):
    """
    Parameters
    ----------
    data_source : {"matminer", "magpie", "deml"}
      Source for element property data.

    """

    self.data_source = data_source

  def _featurize(self, comp):
    """
    Calculate chemical fingerprint from crystal composition.

    Parameters
    ----------
    comp : str
      Reduced formula of crystal.

    Returns
    -------
    feats: np.ndarray
      Vector of properties and statistics derived from chemical
      stoichiometry.

    """

    from pymatgen import Composition
    from matminer.featurizers.composition import ElementProperty

    # Get pymatgen Composition object
    c = Composition(comp)

    ep = ElementProperty.from_preset(self.data_source)

    try:
      feats = ep.featurize(c)
    except:
      feats = []

    return feats


class SineCoulombMatrix(Featurizer):
  """
  Calculate sine Coulomb matrix for crystals.

  A variant of Coulomb matrix for periodic crystals, based on 
  Faber et al. Inter. J. Quantum Chem. 115, 16, (2015).

  The sine Coulomb matrix is identical to the Coulomb matrix, except
  that the inverse distance function is replaced by the inverse of
  sin**2 of the vector between sites which are periodic in the 
  dimensions of the crystal lattice.

  Features are flattened into a vector of matrix eigenvalues by default
  for ML-readiness. To ensure that all feature vectors are equal
  length, the maximum number of atoms (eigenvalues) in the input
  dataset must be specified.

  This featurizer requires the optional dependency pymatgen. It may be
  useful when crystal structures with 3D coordinates are available.

  """

  def __init__(self, max_atoms, eig=True):
    """
    Parameters
    ----------
    max_atoms : int
      Maximum number of atoms for any crystal in the dataset. Used to
      pad the Coulomb matrix.
    eig : bool (default True)
      Return flattened vector of matrix eigenvalues.

    """

    self.max_atoms = int(max_atoms)
    self.eig = eig

  def _featurize(self, struct):
    """
    Calculate sine Coulomb matrix from pymatgen structure.

    Parameters
    ----------
    struct : dict
      Json-serializable dictionary representation of pymatgen.core.structure
      https://pymatgen.org/pymatgen.core.structure.html

    Returns
    -------
    features: np.ndarray
      2D sine Coulomb matrix, or 1D matrix eigenvalues. 

    """

    from pymatgen import Structure

    s = Structure.from_dict(struct)
    features = self.sine_coulomb_matrix(s)
    features = np.asarray(features)

    return features

  def sine_coulomb_matrix(self, s):
    """
    Generate sine Coulomb matrices for each crystal.

    Parameters
    ----------
    s : pymatgen.core.structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    eigs: np.ndarray
      1D matrix eigenvalues. 
    sine_mat: np.ndarray
      2D sine Coulomb matrix.

    """

    sites = s.sites
    atomic_numbers = np.array([site.specie.Z for site in sites])
    sine_mat = np.zeros((len(sites), len(sites)))
    coords = np.array([site.frac_coords for site in sites])
    lattice = s.lattice.matrix

    # Conversion factor
    ang_to_bohr = 1.8897543760313331

    for i in range(len(sine_mat)):
      for j in range(len(sine_mat)):
        if i == j:
          sine_mat[i][i] = 0.5 * atomic_numbers[i]**2.4
        elif i < j:
          vec = coords[i] - coords[j]
          coord_vec = np.sin(np.pi * vec)**2
          trig_dist = np.linalg.norm(
              (np.matrix(coord_vec) * lattice).A1) * ang_to_bohr
          sine_mat[i][j] = atomic_numbers[i] * atomic_numbers[j] / \
                          trig_dist
        else:
          sine_mat[i][j] = sine_mat[j][i]

    if self.eig:  # flatten array to eigenvalues
      eigs, _ = np.linalg.eig(sine_mat)
      zeros = np.zeros((self.max_atoms,))
      zeros[:len(eigs)] = eigs
      eigs = zeros
      return eigs
    else:
      sine_mat = pad_array(sine_mat, self.max_atoms)
      return sine_mat


class StructureGraphFeaturizer(Featurizer):
  """
  Calculate structure graph for crystals.

  Create a graph representation of a crystal structure where atoms
  are nodes and connections between atoms (bonds) are edges. Bonds
  are determined by choosing a strategy for finding nearest neighbors
  from pymatgen.analysis.local_env. For periodic
  graphs, each edge belongs to a lattice image.

  The NetworkX package is used for graph representations.
  Hagberg, A. et al. SciPy2008, 11-15 (2008).

  This featurizer requires the optional dependency pymatgen. It may
  be useful when using graph network models and crystal graph
  convolutional networks.

  #TODO (@ncfrey) process graph features for models

  """

  def __init__(self, strategy=None):
    """
    Parameters
    ----------
    strategy : pymatgen.analysis.local_env.NearNeighbors
      An instance of NearNeighbors that determines how graph is constructed.

    """

    if not strategy:
      from pymatgen.analysis.local_env import MinimumDistanceNN
      strategy = MinimumDistanceNN()

    self.strategy = strategy

  def _featurize(self, struct):
    """
    Calculate structure graph from pymatgen structure.

    Parameters
    ----------
    struct : dict
      Json-serializable dictionary representation of pymatgen.core.structure
      https://pymatgen.org/pymatgen.core.structure.html

    Returns
    -------
    feats: tuple
      atomic numbers, nodes, and edges in networkx.classes.multidigraph.MultiDiGraph format.

    """

    from pymatgen import Structure

    # Get pymatgen structure object
    s = Structure.from_dict(struct)

    features = self._get_structure_graph_features(s)

    return features

  def _get_structure_graph_features(self, struct):
    """
    Calculate structure graph features from pymatgen structure.

    Parameters
    ----------
    struct : pymatgen.core.structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    feats: tuple
      atomic numbers, nodes, and edges in networkx.classes.multidigraph.MultiDiGraph format.
    
    """

    from pymatgen.analysis.graphs import StructureGraph

    atom_features = np.array([site.specie.Z for site in struct], dtype='int32')

    sg = StructureGraph.with_local_env_strategy(struct, self.strategy)
    nodes = np.array(list(sg.graph.nodes))
    edges = np.array(list(sg.graph.edges))

    return (atom_features, nodes, edges)
